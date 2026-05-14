"""
PPO agent for gate library subset selection.

Selects a fixed-size subset of library cell types to optimise area after
technology mapping. Each step swaps one selected cell for one candidate cell.

Usage (from MapTune root):
  python src/ppo_train.py                          # uses config.toml [train]
  python src/ppo_train.py --lib 7nm --bench benchmarks/c880.bench
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
import time
import tomllib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ── paths ─────────────────────────────────────────────────────────────────
_SRC  = os.path.dirname(os.path.abspath(__file__))  # MapTune/src/
_ROOT = os.path.dirname(_SRC)                        # MapTune/
sys.path.insert(0, _SRC)

from abc_mapper import TechMapper, parse_genlib_gates


def _load_config() -> dict:
    with open(os.path.join(_ROOT, "config.toml"), "rb") as f:
        return tomllib.load(f)


def _resolve_library(cfg: dict, lib_name: str) -> str:
    """Resolve a library name (from [train]) to its genlib path."""
    for lib in cfg.get("library", []):
        if lib["name"] == lib_name:
            return os.path.join(_ROOT, lib["genlib"])
    raise ValueError(f"Library '{lib_name}' not found in config.toml [[library]] entries.")

import gymnasium as gym
from gymnasium import spaces

# ═══════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
# Per-cell features extracted from .genlib lines:
#   [area, fanin, rise_block_delay, fall_block_delay]

GATE_DIM = 4

_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\b")
_SKIP   = {"CONST0", "CONST1", "INV", "NONINV", "UNKNOWN"}


def _count_fanin(formula: str) -> int:
    """Count distinct input variable names in a genlib formula."""
    eq = formula.index("=")
    out_var = formula[:eq].strip()
    expr    = formula[eq + 1:].strip()
    if "CONST" in expr:
        return 0
    return len(set(_VAR_RE.findall(expr)) - _SKIP - {out_var})


def extract_cell_features(gate_lines: List[str]) -> np.ndarray:
    """Return (N, GATE_DIM) feature matrix from genlib GATE lines.

    genlib line format:
      GATE <name> <area> <formula>; PIN <name> <phase> <in_load> <max_load>
           <rise_blk> <rise_fan> <fall_blk> <fall_fan>
    """
    rows = []
    for line in gate_lines:
        parts = line.split()
        area  = float(parts[2])

        after_area = line[line.index(parts[2]) + len(parts[2]):].strip()
        semi_idx   = after_area.index(";")
        formula    = after_area[:semi_idx].strip()
        fanin      = float(_count_fanin(formula))

        rise_blk = fall_blk = 0.0
        if "PIN" in line:
            pin_parts = line[line.index("PIN"):].split()
            # PIN name phase in_load max_load rise_blk rise_fan fall_blk fall_fan
            if len(pin_parts) >= 9:
                try:
                    rise_blk = float(pin_parts[5])
                    fall_blk = float(pin_parts[7])
                except ValueError:
                    pass

        rows.append([area, fanin, rise_blk, fall_blk])

    return np.array(rows, dtype=np.float32)


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std:  Optional[np.ndarray] = None,
    eps:  float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = features.mean(axis=0)
        std  = features.std(axis=0)
    return (features - mean) / (std + eps), mean, std


# ═══════════════════════════════════════════════════════════════════════════
# 2. NETLIST PARSER  (.bench)
# ═══════════════════════════════════════════════════════════════════════════
# Parses ISCAS-format bench files to compute global circuit context features.
# Context vector (CONTEXT_DIM = 6):
#   [log(num_gates), log(max_depth), mean_fanin/4, mean_fanout/4,
#    log(num_pis), log(num_pos)]

CONTEXT_DIM = 6


@dataclass
class _Gate:
    name: str
    gate_type: str
    inputs: List[str] = field(default_factory=list)


@dataclass
class Circuit:
    primary_inputs: List[str] = field(default_factory=list)
    primary_outputs: List[str] = field(default_factory=list)
    gates: Dict[str, _Gate] = field(default_factory=dict)
    fanout: Dict[str, List[str]] = field(default_factory=dict)
    depth: Dict[str, int] = field(default_factory=dict)


def parse_bench(path: str) -> Circuit:
    circuit = Circuit()
    gate_re = re.compile(r"^(\S+)\s*=\s*(\w+)\s*\(([^)]*)\)")

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("INPUT("):
                circuit.primary_inputs.append(line[6:-1].strip())
            elif line.startswith("OUTPUT("):
                circuit.primary_outputs.append(line[7:-1].strip())
            else:
                m = gate_re.match(line)
                if m:
                    out = m.group(1).strip()
                    gtype = m.group(2).strip()
                    ins = [x.strip() for x in m.group(3).split(",") if x.strip()]
                    circuit.gates[out] = _Gate(out, gtype, ins)

    # Fanout map
    fanout: Dict[str, List[str]] = defaultdict(list)
    for name, gate in circuit.gates.items():
        for inp in gate.inputs:
            fanout[inp].append(name)
    circuit.fanout = dict(fanout)

    # Logic depth via iterative relaxation
    depth: Dict[str, int] = {}
    for pi in circuit.primary_inputs:
        depth[pi] = 0
    # DFF outputs are pseudo-PIs
    for name, gate in circuit.gates.items():
        if gate.gate_type.upper() == "DFF":
            depth[name] = 0

    remaining = {n: g for n, g in circuit.gates.items()
                 if n not in depth and g.gate_type.upper() != "DFF"}
    changed = True
    while changed:
        changed = False
        for name, gate in list(remaining.items()):
            if all(inp in depth for inp in gate.inputs):
                depth[name] = max((depth[inp] for inp in gate.inputs), default=0) + 1
                del remaining[name]
                changed = True
    for name in remaining:
        depth[name] = 0

    circuit.depth = depth
    return circuit


def compute_context(circuit: Circuit) -> np.ndarray:
    import math
    n_gates  = len(circuit.gates)
    depths   = list(circuit.depth.values())
    fanins   = [len(g.inputs) for g in circuit.gates.values()]
    fanouts  = [len(circuit.fanout.get(n, [])) for n in circuit.gates]

    max_depth   = float(max(depths))   if depths  else 0.0
    mean_fanin  = float(np.mean(fanins))  if fanins  else 0.0
    mean_fanout = float(np.mean(fanouts)) if fanouts else 0.0

    ctx = np.array([
        math.log1p(n_gates) / 10.0,
        math.log1p(max_depth) / 5.0,
        mean_fanin  / 4.0,
        mean_fanout / 4.0,
        math.log1p(len(circuit.primary_inputs))  / 4.0,
        math.log1p(len(circuit.primary_outputs)) / 4.0,
    ], dtype=np.float32)
    return np.clip(ctx, -5.0, 5.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. GYMNASIUM ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════
# State  : {"selected": (N_SEL, GATE_DIM),
#            "candidates": (K, GATE_DIM),
#            "context": (CONTEXT_DIM + 4,)}
# Action : [remove_idx, add_idx]  — one swap per step
# Reward : Δarea improvement (positive = better)

TOTAL_CONTEXT_DIM = CONTEXT_DIM + 4   # circuit-static + dynamic


class CircuitEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        mapper: TechMapper,
        norm_features: np.ndarray,   # (N_total, GATE_DIM) already normalised
        circuit: Circuit,
        n_select: int = 50,
        max_steps: int = 200,
        tag: str = "ppo",
    ):
        super().__init__()
        self.mapper        = mapper
        self.all_feats     = norm_features
        self.circuit_ctx   = compute_context(circuit)
        self.n_total       = len(norm_features)
        self.n_select      = n_select
        self.n_cand        = self.n_total - n_select
        self.max_steps     = max_steps
        self.tag           = tag

        self.action_space = spaces.MultiDiscrete([n_select, self.n_cand])
        self.observation_space = spaces.Dict({
            "selected":   spaces.Box(-np.inf, np.inf, (n_select,  GATE_DIM), np.float32),
            "candidates": spaces.Box(-np.inf, np.inf, (self.n_cand, GATE_DIM), np.float32),
            "context":    spaces.Box(-np.inf, np.inf, (TOTAL_CONTEXT_DIM,),    np.float32),
        })

        self._sel: List[int]  = []
        self._cand: List[int] = []
        self._step     = 0
        self._last_area = mapper.baseline_area
        self._best_area = float("inf")

    # ── Gym API ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        perm = np.random.permutation(self.n_total)
        self._sel  = sorted(perm[:self.n_select].tolist())
        self._cand = sorted(perm[self.n_select:].tolist())
        self._step = 0

        area, delay = self.mapper.map_subset(self._sel, self.tag)
        self._last_area = area if np.isfinite(area) else self.mapper.baseline_area
        self._best_area = self._last_area
        return self._obs(self._last_area, delay), {}

    def step(self, action):
        r_idx, a_idx = int(action[0]), int(action[1])

        removed = self._sel[r_idx]
        added   = self._cand[a_idx]
        self._sel[r_idx]   = added
        self._cand[a_idx]  = removed
        self._sel.sort()
        self._cand.sort()
        self._step += 1

        area, delay = self.mapper.map_subset(self._sel, self.tag)
        if not np.isfinite(area):
            area = self._last_area

        reward = (self._last_area - area) / (self.mapper.baseline_area + 1e-8)
        self._last_area = area
        self._best_area = min(self._best_area, area)
        done = self._step >= self.max_steps

        info = {"area": area, "delay": delay, "best_area": self._best_area}
        return self._obs(area, delay), reward, done, False, info

    def render(self, mode="human"):
        print(f"Step {self._step}/{self.max_steps}  area={self._last_area:.3f}")

    def close(self):
        pass

    # ── Internal ──────────────────────────────────────────────────────────

    def _obs(self, last_area: float, last_delay: float) -> dict:
        sel_f  = self.all_feats[self._sel]   # (N_sel, d)
        cand_f = self.all_feats[self._cand]  # (K, d)
        ba, bd = self.mapper.baseline_area, self.mapper.baseline_delay
        dyn = np.array([
            last_area  / (ba + 1e-8),
            last_delay / (bd + 1e-8),
            self._step / self.max_steps,
            self.n_select / self.n_total,
        ], dtype=np.float32)
        ctx = np.concatenate([self.circuit_ctx, dyn])
        return {"selected": sel_f, "candidates": cand_f, "context": ctx}


# ═══════════════════════════════════════════════════════════════════════════
# 4. ACTOR  (pointer / attention policy)
# ═══════════════════════════════════════════════════════════════════════════
# Architecture:
#   Encoder : mean-pool(selected) + context → h  (B, H)
#   Score_i = v^T tanh(W_gate * embed(f_i) + W_ctx * h)
#   Separate v_remove / v_add heads; softmax → Categorical

HIDDEN   = 128
ATTN_DIM = 64


class PointerActor(nn.Module):
    def __init__(self, gate_dim: int = GATE_DIM, ctx_dim: int = TOTAL_CONTEXT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.gate_embed = nn.Sequential(
            nn.Linear(gate_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_proj  = nn.Linear(ctx_dim, hidden)
        self.pool_proj = nn.Linear(hidden * 2, hidden)

        self.W_gate   = nn.Linear(hidden, ATTN_DIM, bias=False)
        self.W_ctx    = nn.Linear(hidden, ATTN_DIM, bias=False)
        self.v_remove = nn.Linear(ATTN_DIM, 1, bias=False)
        self.v_add    = nn.Linear(ATTN_DIM, 1, bias=False)

    def _encode(self, sel: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # sel: (B,50,d)  ctx: (B,ctx_dim)  →  h: (B,H)
        pooled  = self.gate_embed(sel).mean(dim=1)            # (B,H)
        ctx_h   = self.ctx_proj(ctx)                          # (B,H)
        return F.relu(self.pool_proj(torch.cat([pooled, ctx_h], -1)))  # (B,H)

    def _score(self, feats: torch.Tensor, h: torch.Tensor, v: nn.Module) -> torch.Tensor:
        # feats: (B,N,d)  →  logits: (B,N)
        emb     = self.gate_embed(feats)                      # (B,N,H)
        W_gate  = self.W_gate(emb)                            # (B,N,A)
        W_ctx   = self.W_ctx(h).unsqueeze(1)                  # (B,1,A)
        return v(torch.tanh(W_gate + W_ctx)).squeeze(-1)      # (B,N)

    def forward(self, sel, cand, ctx):
        h = self._encode(sel, ctx)
        return self._score(sel, h, self.v_remove), self._score(cand, h, self.v_add)

    @torch.no_grad()
    def get_action(self, sel, cand, ctx):
        rl, al = self.forward(sel, cand, ctx)
        dr, da = Categorical(logits=rl), Categorical(logits=al)
        r, a   = dr.sample(), da.sample()
        lp     = dr.log_prob(r) + da.log_prob(a)
        ent    = dr.entropy()   + da.entropy()
        return r, a, lp, ent

    def evaluate_actions(self, sel, cand, ctx, r_acts, a_acts):
        rl, al = self.forward(sel, cand, ctx)
        dr, da = Categorical(logits=rl), Categorical(logits=al)
        lp     = dr.log_prob(r_acts) + da.log_prob(a_acts)
        ent    = dr.entropy()        + da.entropy()
        return lp, ent


# ═══════════════════════════════════════════════════════════════════════════
# 5. CRITIC  (set encoder → scalar value)
# ═══════════════════════════════════════════════════════════════════════════

class SetCritic(nn.Module):
    def __init__(self, gate_dim: int = GATE_DIM, ctx_dim: int = TOTAL_CONTEXT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(gate_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, sel: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        pooled  = self.embed(sel).mean(dim=1)      # (B,H)
        ctx_h   = self.ctx_proj(ctx)               # (B,H)
        return self.mlp(torch.cat([pooled, ctx_h], -1)).squeeze(-1)  # (B,)


# ═══════════════════════════════════════════════════════════════════════════
# 6. ROLLOUT BUFFER
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self, capacity: int, n_sel: int, n_cand: int):
        self.capacity = capacity
        self.sel_feats  = np.zeros((capacity, n_sel,  GATE_DIM), np.float32)
        self.cand_feats = np.zeros((capacity, n_cand, GATE_DIM), np.float32)
        self.ctx        = np.zeros((capacity, TOTAL_CONTEXT_DIM), np.float32)
        self.r_acts     = np.zeros(capacity, np.int64)
        self.a_acts     = np.zeros(capacity, np.int64)
        self.log_probs  = np.zeros(capacity, np.float32)
        self.rewards    = np.zeros(capacity, np.float32)
        self.values     = np.zeros(capacity, np.float32)
        self.dones      = np.zeros(capacity, np.float32)
        self.ptr = self.size = 0

    def push(self, sel, cand, ctx, r_act, a_act, lp, reward, value, done):
        i = self.ptr
        self.sel_feats[i]  = sel
        self.cand_feats[i] = cand
        self.ctx[i]        = ctx
        self.r_acts[i]     = r_act
        self.a_acts[i]     = a_act
        self.log_probs[i]  = lp
        self.rewards[i]    = reward
        self.values[i]     = value
        self.dones[i]      = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, lam: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        n   = self.size
        adv = np.zeros(n, np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nv    = last_value if t == n - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * nv * (1 - self.dones[t]) - self.values[t]
            gae   = delta + gamma * lam * (1 - self.dones[t]) * gae
            adv[t] = gae
        ret = adv + self.values[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def tensors(self, device: str):
        n = self.size
        t = lambda x: torch.from_numpy(x[:n]).to(device)
        return (
            t(self.sel_feats), t(self.cand_feats), t(self.ctx),
            t(self.r_acts).long(), t(self.a_acts).long(),
            t(self.log_probs),
        )

    def clear(self):
        self.ptr = self.size = 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. PPO TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    def __init__(
        self,
        n_select: int,
        n_candidates: int,
        lr: float          = 3e-4,
        gamma: float       = 0.99,
        gae_lambda: float  = 0.95,
        clip_eps: float    = 0.2,
        value_coef: float  = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int      = 4,
        batch_size: int    = 64,
        rollout_len: int   = 200,
        device: str        = None,
    ):
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.clip_eps    = clip_eps
        self.value_coef  = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size

        self.actor  = PointerActor().to(self.device)
        self.critic = SetCritic().to(self.device)
        self.opt    = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.buffer = RolloutBuffer(rollout_len, n_select, n_candidates)

    @torch.no_grad()
    def act(self, obs: dict):
        sel  = torch.FloatTensor(obs["selected"]).unsqueeze(0).to(self.device)
        cand = torch.FloatTensor(obs["candidates"]).unsqueeze(0).to(self.device)
        ctx  = torch.FloatTensor(obs["context"]).unsqueeze(0).to(self.device)
        r, a, lp, _ = self.actor.get_action(sel, cand, ctx)
        v           = self.critic(sel, ctx)
        return r.item(), a.item(), lp.item(), v.item()

    def store(self, obs, r_act, a_act, lp, reward, value, done):
        self.buffer.push(
            obs["selected"], obs["candidates"], obs["context"],
            r_act, a_act, lp, reward, value, float(done),
        )

    def update(self, last_value: float = 0.0) -> dict:
        adv, ret = self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        adv_t = torch.FloatTensor(adv).to(self.device)
        ret_t = torch.FloatTensor(ret).to(self.device)

        sel, cand, ctx, r_acts, a_acts, old_lp = self.buffer.tensors(self.device)
        n = self.buffer.size
        logs = {"pi_loss": [], "v_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s: s + self.batch_size]
                new_lp, ent = self.actor.evaluate_actions(
                    sel[idx], cand[idx], ctx[idx], r_acts[idx], a_acts[idx]
                )
                v = self.critic(sel[idx], ctx[idx])

                ratio = torch.exp(new_lp - old_lp[idx])
                b_adv = adv_t[idx]
                pi_loss = -torch.min(
                    ratio * b_adv,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv,
                ).mean()
                v_loss  = F.mse_loss(v, ret_t[idx])
                e_loss  = -ent.mean()

                loss = pi_loss + self.value_coef * v_loss + self.entropy_coef * e_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.opt.step()

                logs["pi_loss"].append(pi_loss.item())
                logs["v_loss"].append(v_loss.item())
                logs["entropy"].append(-e_loss.item())

        self.buffer.clear()
        return {k: float(np.mean(v)) for k, v in logs.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 8. LOGGING & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "episode", "reward", "area", "area_norm",
    "best_area", "best_area_norm", "delay",
    "pi_loss", "v_loss", "entropy", "elapsed_s",
]

_SMOOTH = 20   # rolling-average window for reward plot


class RunLogger:
    """Creates a timestamped run directory and writes CSV + plots."""

    def __init__(self, run_dir: str, baseline_area: float, baseline_delay: float, cfg_dict: dict):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir       = run_dir
        self.baseline_area  = baseline_area
        self.baseline_delay = baseline_delay

        # persist run config
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

        self._csv_path = os.path.join(run_dir, "training.csv")
        self._csv_f    = open(self._csv_path, "w", newline="")
        self._writer   = csv.DictWriter(self._csv_f, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()

        # in-memory history for plotting
        self.history: Dict[str, List[float]] = {k: [] for k in _CSV_FIELDS}

    def log(self, episode: int, reward: float, area: float, delay: float,
            best_area: float, metrics: dict, elapsed: float):
        row = {
            "episode":       episode,
            "reward":        round(reward, 6),
            "area":          round(area, 4),
            "area_norm":     round(area / (self.baseline_area + 1e-8), 6),
            "best_area":     round(best_area, 4),
            "best_area_norm": round(best_area / (self.baseline_area + 1e-8), 6),
            "delay":         round(delay, 4),
            "pi_loss":       round(metrics["pi_loss"], 6),
            "v_loss":        round(metrics["v_loss"], 6),
            "entropy":       round(metrics["entropy"], 6),
            "elapsed_s":     round(elapsed, 1),
        }
        self._writer.writerow(row)
        self._csv_f.flush()
        for k, v in row.items():
            self.history[k].append(v)

    def finalize(self):
        self._csv_f.close()
        self._save_plots()
        print(f"\nLogs  → {self._csv_path}")
        print(f"Plots → {self.run_dir}/")

    # ── plot helpers ──────────────────────────────────────────────────────

    def _rolling(self, values: List[float], w: int) -> List[float]:
        out = []
        for i in range(len(values)):
            s = max(0, i - w + 1)
            out.append(float(np.mean(values[s: i + 1])))
        return out

    def _save_plots(self):
        eps = self.history["episode"]
        if not eps:
            return

        # ── 1. reward ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        rewards = self.history["reward"]
        ax.plot(eps, rewards, alpha=0.3, color="steelblue", linewidth=0.8, label="reward")
        ax.plot(eps, self._rolling(rewards, _SMOOTH),
                color="steelblue", linewidth=1.8, label=f"rolling mean ({_SMOOTH} ep)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("Episode reward")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "reward.png"), dpi=150)
        plt.close(fig)

        # ── 2. area ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        area_norm  = self.history["area_norm"]
        best_norm  = self.history["best_area_norm"]
        ax.plot(eps, area_norm, alpha=0.35, color="coral", linewidth=0.8, label="area (norm)")
        ax.plot(eps, self._rolling(area_norm, _SMOOTH),
                color="coral", linewidth=1.8, label=f"rolling mean ({_SMOOTH} ep)")
        ax.plot(eps, best_norm, color="darkred", linewidth=1.4,
                linestyle="--", label="best area so far")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", label="baseline (1.0)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Area / baseline")
        ax.set_title("Normalised area (lower = better)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "area.png"), dpi=150)
        plt.close(fig)

        # ── 3. losses ──────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for ax, key, color, label in zip(
            axes,
            ["pi_loss", "v_loss", "entropy"],
            ["royalblue", "seagreen", "darkorange"],
            ["Policy loss (π)", "Value loss (V)", "Entropy (H)"],
        ):
            vals = self.history[key]
            ax.plot(eps, vals, alpha=0.35, color=color, linewidth=0.8)
            ax.plot(eps, self._rolling(vals, _SMOOTH),
                    color=color, linewidth=1.8)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        axes[-1].set_xlabel("Episode")
        fig.suptitle("PPO losses")
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "losses.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 9. TRAINING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    cfg   = _load_config()
    train = cfg.get("train", {})

    p = argparse.ArgumentParser()
    p.add_argument("--lib",       default=train.get("library",   "nan45"),
                   help="Library name from config.toml [[library]] (e.g. nan45, 7nm)")
    p.add_argument("--bench",     default=train.get("benchmark", "benchmarks/s838a.bench"),
                   help="Path to .bench / .blif file (relative to MapTune root)")
    p.add_argument("--n-select",  type=int,   default=train.get("n_select",  50))
    p.add_argument("--max-steps", type=int,   default=train.get("max_steps", 200))
    p.add_argument("--episodes",  type=int,   default=train.get("episodes",  300))
    p.add_argument("--lr",        type=float, default=train.get("lr",        3e-4))
    p.add_argument("--seed",      type=int,   default=train.get("seed",      42))
    return p.parse_args(), cfg


def main():
    args, cfg = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    genlib_path = _resolve_library(cfg, args.lib)
    bench_path  = os.path.join(_ROOT, args.bench)
    gen_dir     = os.path.join(_ROOT, "gen_newlibs") + os.sep
    bench_stem  = os.path.splitext(os.path.basename(bench_path))[0]
    temp_blif   = os.path.join(_ROOT, "temp_blifs", f"{bench_stem}_ppo_temp.blif")

    os.makedirs(os.path.join(_ROOT, "temp_blifs"),  exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "gen_newlibs"), exist_ok=True)

    print(f"Library  : {args.lib}  ({genlib_path})")
    print(f"Benchmark: {bench_path}")

    mapper = TechMapper(genlib_path, bench_path, gen_dir, temp_blif, area_mode=True)
    print(f"Baseline — Area: {mapper.baseline_area:.3f}  Delay: {mapper.baseline_delay:.3f} ps")
    print(f"Mutable gates  : {mapper.num_arms}")

    if mapper.num_arms < args.n_select:
        raise ValueError(
            f"Only {mapper.num_arms} mutable gates available; "
            f"--n-select {args.n_select} is too large."
        )

    raw_feats  = extract_cell_features(mapper.mutable_gates)
    norm_feats, _, _ = normalize_features(raw_feats)
    circuit = parse_bench(bench_path)

    env = CircuitEnv(
        mapper, norm_feats, circuit,
        n_select=args.n_select, max_steps=args.max_steps,
    )

    n_cand = mapper.num_arms - args.n_select
    agent  = PPOTrainer(
        n_select=args.n_select, n_candidates=n_cand,
        lr=args.lr, rollout_len=args.max_steps,
    )

    # ── run directory: logs/{lib}_{bench}_{timestamp} ─────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{args.lib}_{bench_stem}_{timestamp}"
    run_dir   = os.path.join(_ROOT, "logs", run_name)
    logger = RunLogger(
        run_dir,
        baseline_area=mapper.baseline_area,
        baseline_delay=mapper.baseline_delay,
        cfg_dict=vars(args),
    )
    print(f"Run dir  : {run_dir}\n")

    best_area = float("inf")
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0

        for _ in range(args.max_steps):
            r_idx, a_idx, lp, value = agent.act(obs)
            next_obs, reward, done, _, info = env.step(np.array([r_idx, a_idx]))
            agent.store(obs, r_idx, a_idx, lp, reward, value, done)
            ep_reward += reward
            obs = next_obs
            if done:
                break

        # Bootstrap value for GAE
        with torch.no_grad():
            sel = torch.FloatTensor(obs["selected"]).unsqueeze(0).to(agent.device)
            ctx = torch.FloatTensor(obs["context"]).unsqueeze(0).to(agent.device)
            last_val = agent.critic(sel, ctx).item()

        metrics  = agent.update(last_value=last_val)
        elapsed  = time.time() - t0

        if info["area"] < best_area:
            best_area = info["area"]

        logger.log(ep, ep_reward, info["area"], info["delay"], best_area, metrics, elapsed)

        if ep % 10 == 0:
            print(
                f"Ep {ep:4d}/{args.episodes}  "
                f"reward {ep_reward:+.4f}  "
                f"area {info['area']:.3f}  "
                f"best {best_area:.3f}  "
                f"π {metrics['pi_loss']:.4f}  "
                f"V {metrics['v_loss']:.4f}  "
                f"H {metrics['entropy']:.4f}  "
                f"[{elapsed:.0f}s]"
            )

    logger.finalize()

    improvement = (mapper.baseline_area - best_area) / mapper.baseline_area * 100
    print(f"\nDone.  Best area {best_area:.3f} / baseline {mapper.baseline_area:.3f}  "
          f"({improvement:+.1f}%)")


if __name__ == "__main__":
    main()
