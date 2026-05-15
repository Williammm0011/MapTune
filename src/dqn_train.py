"""
DQN agent for gate library subset selection.

Uses a Deep Q-Network with experience replay and a target network.
Same environment as ppo_train.py: each step toggles one gate in/out of the
selected subset, optimising area after technology mapping.

Usage (from MapTune root):
  python src/dqn_train.py                          # uses config.toml [train]
  python src/dqn_train.py --lib 7nm --bench benchmarks/c880.bench
"""

import gymnasium as gym
from gymnasium import spaces
from abc_mapper import TechMapper, parse_genlib_gates
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import datetime
import json
import os
import re
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")


# ── paths ─────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)


def _load_config() -> dict:
    with open(os.path.join(_ROOT, "config.toml"), "rb") as f:
        return tomllib.load(f)


def _resolve_library(cfg: dict, lib_name: str) -> str:
    for lib in cfg.get("library", []):
        if lib["name"] == lib_name:
            return os.path.join(_ROOT, lib["genlib"])
    raise ValueError(
        f"Library '{lib_name}' not found in config.toml [[library]] entries.")


# ═══════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
GATE_DIM = 4

_VAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\b")
_SKIP = {"CONST0", "CONST1", "INV", "NONINV", "UNKNOWN"}


def _count_fanin(formula: str) -> int:
    eq = formula.index("=")
    out_var = formula[:eq].strip()
    expr = formula[eq + 1:].strip()
    if "CONST" in expr:
        return 0
    return len(set(_VAR_RE.findall(expr)) - _SKIP - {out_var})


def extract_cell_features(gate_lines: List[str]) -> np.ndarray:
    rows = []
    for line in gate_lines:
        parts = line.split()
        area = float(parts[2])

        after_area = line[line.index(parts[2]) + len(parts[2]):].strip()
        semi_idx = after_area.index(";")
        formula = after_area[:semi_idx].strip()
        fanin = float(_count_fanin(formula))

        rise_blk = fall_blk = 0.0
        if "PIN" in line:
            pin_parts = line[line.index("PIN"):].split()
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
        std = features.std(axis=0)
    return (features - mean) / (std + eps), mean, std


# ═══════════════════════════════════════════════════════════════════════════
# 2. NETLIST PARSER  (.bench)
# ═══════════════════════════════════════════════════════════════════════════
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
                    ins = [x.strip()
                           for x in m.group(3).split(",") if x.strip()]
                    circuit.gates[out] = _Gate(out, gtype, ins)

    fanout: Dict[str, List[str]] = defaultdict(list)
    for name, gate in circuit.gates.items():
        for inp in gate.inputs:
            fanout[inp].append(name)
    circuit.fanout = dict(fanout)

    depth: Dict[str, int] = {}
    for pi in circuit.primary_inputs:
        depth[pi] = 0
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
                depth[name] = max((depth[inp]
                                  for inp in gate.inputs), default=0) + 1
                del remaining[name]
                changed = True
    for name in remaining:
        depth[name] = 0

    circuit.depth = depth
    return circuit


def compute_context(circuit: Circuit) -> np.ndarray:
    import math
    n_gates = len(circuit.gates)
    depths = list(circuit.depth.values())
    fanins = [len(g.inputs) for g in circuit.gates.values()]
    fanouts = [len(circuit.fanout.get(n, [])) for n in circuit.gates]

    max_depth = float(max(depths)) if depths else 0.0
    mean_fanin = float(np.mean(fanins)) if fanins else 0.0
    mean_fanout = float(np.mean(fanouts)) if fanouts else 0.0

    ctx = np.array([
        math.log1p(n_gates) / 10.0,
        math.log1p(max_depth) / 5.0,
        mean_fanin / 4.0,
        mean_fanout / 4.0,
        math.log1p(len(circuit.primary_inputs)) / 4.0,
        math.log1p(len(circuit.primary_outputs)) / 4.0,
    ], dtype=np.float32)
    return np.clip(ctx, -5.0, 5.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. GYMNASIUM ENVIRONMENT  (identical to PPO version)
# ═══════════════════════════════════════════════════════════════════════════
TOTAL_CONTEXT_DIM = CONTEXT_DIM + 4


class CircuitEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        mapper: TechMapper,
        norm_features: np.ndarray,
        circuit: Circuit,
        n_select: int = 50,
        max_steps: int = 500,
        tag: str = "dqn",
    ):
        super().__init__()
        self.mapper = mapper
        self.all_feats = norm_features
        self.circuit_ctx = compute_context(circuit)
        self.n_total = len(norm_features)
        self.n_select_init = n_select
        self.max_steps = max_steps
        self.tag = tag

        self.action_space = spaces.Discrete(self.n_total)
        self.observation_space = spaces.Dict({
            "gate_feats":    spaces.Box(-np.inf, np.inf, (self.n_total, GATE_DIM), np.float32),
            "selected_mask": spaces.Box(0.0, 1.0,        (self.n_total,),          np.float32),
            "context":       spaces.Box(-np.inf, np.inf, (TOTAL_CONTEXT_DIM,),     np.float32),
        })

        self._selected: set = set()
        self._step = 0
        self._last_area = mapper.baseline_area
        self._last_delay = mapper.baseline_delay
        self._best_area = float("inf")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        perm = np.random.permutation(self.n_total)
        self._selected = set(perm[:self.n_select_init].tolist())
        self._step = 0

        area, delay = self.mapper.map_subset(sorted(self._selected), self.tag)
        self._last_area = area if np.isfinite(
            area) else self.mapper.baseline_area
        self._last_delay = delay
        self._best_area = self._last_area
        return self._obs(), {}

    def step(self, action):
        gate_idx = int(action)

        if gate_idx in self._selected:
            self._selected.discard(gate_idx)
        else:
            self._selected.add(gate_idx)
        self._step += 1

        sel_list = sorted(self._selected)
        if sel_list:
            area, delay = self.mapper.map_subset(sel_list, self.tag)
            if not np.isfinite(area):
                area, delay = self._last_area, self._last_delay
        else:
            area, delay = self._last_area, self._last_delay

        reward = (self._last_area - area) / (self.mapper.baseline_area + 1e-8)
        self._last_area = area
        self._last_delay = delay
        self._best_area = min(self._best_area, area)
        done = self._step >= self.max_steps

        info = {"area": area, "delay": delay,
                "best_area": self._best_area, "n_selected": len(self._selected)}
        return self._obs(), reward, done, False, info

    def render(self, mode="human"):
        print(f"Step {self._step}/{self.max_steps}  "
              f"area={self._last_area:.3f}  n_sel={len(self._selected)}")

    def close(self):
        pass

    def _obs(self) -> dict:
        mask = np.zeros(self.n_total, dtype=np.float32)
        for idx in self._selected:
            mask[idx] = 1.0
        ba, bd = self.mapper.baseline_area, self.mapper.baseline_delay
        dyn = np.array([
            self._last_area / (ba + 1e-8),
            self._last_delay / (bd + 1e-8),
            self._step / self.max_steps,
            len(self._selected) / self.n_total,
        ], dtype=np.float32)
        ctx = np.concatenate([self.circuit_ctx, dyn])
        return {"gate_feats": self.all_feats, "selected_mask": mask, "context": ctx}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Q-NETWORK  (attention-based, outputs one Q-value per gate action)
# ═══════════════════════════════════════════════════════════════════════════
# Architecture mirrors GateActor from ppo_train.py but predicts Q(s, a)
# for every gate a simultaneously:
#   Tag gate with is_selected, embed → per-gate embeddings
#   Masked mean-pool selected gates + context → global context h
#   Q_i = v^T tanh(W_gate * embed(f_i) + W_ctx * h)

HIDDEN = 128
ATTN_DIM = 64


class GateQNetwork(nn.Module):
    def __init__(self, gate_dim: int = GATE_DIM, ctx_dim: int = TOTAL_CONTEXT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.gate_embed = nn.Sequential(
            nn.Linear(gate_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_proj = nn.Linear(ctx_dim, hidden)
        self.pool_proj = nn.Linear(hidden * 2, hidden)

        self.W_gate = nn.Linear(hidden, ATTN_DIM, bias=False)
        self.W_ctx  = nn.Linear(hidden, ATTN_DIM, bias=False)
        self.v      = nn.Linear(ATTN_DIM, 1, bias=False)

    def forward(
        self,
        gate_feats: torch.Tensor,  # (B, N, gate_dim)
        mask: torch.Tensor,        # (B, N)
        ctx: torch.Tensor,         # (B, ctx_dim)
    ) -> torch.Tensor:             # (B, N)  Q-values per gate
        tagged = torch.cat([gate_feats, mask.unsqueeze(-1)], dim=-1)  # (B,N,d+1)
        emb = self.gate_embed(tagged)                                   # (B,N,H)

        n_sel = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / n_sel          # (B,H)
        ctx_h  = self.ctx_proj(ctx)                                     # (B,H)
        h = F.relu(self.pool_proj(torch.cat([pooled, ctx_h], -1)))      # (B,H)

        W_gate = self.W_gate(emb)              # (B,N,A)
        W_ctx  = self.W_ctx(h).unsqueeze(1)   # (B,1,A)
        q = self.v(torch.tanh(W_gate + W_ctx)).squeeze(-1)  # (B,N)
        return q


# ═══════════════════════════════════════════════════════════════════════════
# 5. REPLAY BUFFER  (off-policy experience replay)
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int, n_total: int):
        self.capacity = capacity
        self.n_total = n_total
        self.gate_feats  = np.zeros((capacity, n_total, GATE_DIM),        np.float32)
        self.masks       = np.zeros((capacity, n_total),                   np.float32)
        self.ctx         = np.zeros((capacity, TOTAL_CONTEXT_DIM),         np.float32)
        self.next_gate_feats = np.zeros((capacity, n_total, GATE_DIM),    np.float32)
        self.next_masks  = np.zeros((capacity, n_total),                   np.float32)
        self.next_ctx    = np.zeros((capacity, TOTAL_CONTEXT_DIM),         np.float32)
        self.actions     = np.zeros(capacity, np.int64)
        self.rewards     = np.zeros(capacity, np.float32)
        self.dones       = np.zeros(capacity, np.float32)
        self.ptr = self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.gate_feats[i]      = obs["gate_feats"]
        self.masks[i]           = obs["selected_mask"]
        self.ctx[i]             = obs["context"]
        self.next_gate_feats[i] = next_obs["gate_feats"]
        self.next_masks[i]      = next_obs["selected_mask"]
        self.next_ctx[i]        = next_obs["context"]
        self.actions[i]         = action
        self.rewards[i]         = reward
        self.dones[i]           = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str):
        idx = np.random.randint(0, self.size, batch_size)
        def t(x): return torch.from_numpy(x[idx]).to(device)
        return (
            t(self.gate_feats), t(self.masks), t(self.ctx),
            t(self.actions).long(),
            t(self.rewards),
            t(self.next_gate_feats), t(self.next_masks), t(self.next_ctx),
            t(self.dones),
        )

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════
# 6. DQN TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class DQNTrainer:
    def __init__(
        self,
        n_total: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        target_update_freq: int = 200,   # hard target-network update every N gradient steps
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 5_000,
        min_buffer: int = 512,           # start learning only once buffer has this many samples
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.min_buffer = min_buffer

        self.online_net = GateQNetwork().to(self.device)
        self.target_net = GateQNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.opt = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, n_total)

        self._step = 0   # gradient steps taken

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs: dict) -> int:
        if np.random.random() < self.epsilon:
            n = obs["gate_feats"].shape[0]
            return int(np.random.randint(n))

        gf   = torch.FloatTensor(obs["gate_feats"]).unsqueeze(0).to(self.device)
        mask = torch.FloatTensor(obs["selected_mask"]).unsqueeze(0).to(self.device)
        ctx  = torch.FloatTensor(obs["context"]).unsqueeze(0).to(self.device)
        q = self.online_net(gf, mask, ctx)   # (1, N)
        return int(q.argmax(dim=-1).item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> Optional[dict]:
        if len(self.buffer) < self.min_buffer:
            return None

        gf, masks, ctx, actions, rewards, ngf, nmasks, nctx, dones = \
            self.buffer.sample(self.batch_size, self.device)

        # Current Q-values for chosen actions
        q_all    = self.online_net(gf, masks, ctx)              # (B, N)
        q_chosen = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN: online net selects next action, target net evaluates it
        with torch.no_grad():
            next_actions = self.online_net(ngf, nmasks, nctx).argmax(dim=-1)  # (B,)
            next_q = self.target_net(ngf, nmasks, nctx)                        # (B, N)
            next_q_max = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)
            target = rewards + self.gamma * next_q_max * (1.0 - dones)

        loss = F.smooth_l1_loss(q_chosen, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.opt.step()

        self._step += 1

        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "q_loss":  loss.item(),
            "epsilon": self.epsilon,
            "mean_q":  float(q_all.mean().item()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. LOGGING & PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "episode", "reward", "area", "area_norm",
    "best_area", "best_area_norm", "delay", "n_selected",
    "q_loss", "epsilon", "mean_q", "elapsed_s",
]

_SMOOTH = 20


class RunLogger:
    def __init__(self, run_dir: str, baseline_area: float, baseline_delay: float, cfg_dict: dict):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.baseline_area = baseline_area
        self.baseline_delay = baseline_delay

        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

        self._csv_path = os.path.join(run_dir, "training.csv")
        self._csv_f    = open(self._csv_path, "w", newline="")
        self._writer   = csv.DictWriter(self._csv_f, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()

        self.history: Dict[str, List[float]] = {k: [] for k in _CSV_FIELDS}

    def log(self, episode: int, reward: float, area: float, delay: float,
            n_selected: int, best_area: float, metrics: dict, elapsed: float):
        row = {
            "episode":        episode,
            "reward":         round(reward, 6),
            "area":           round(area, 4),
            "area_norm":      round(area / (self.baseline_area + 1e-8), 6),
            "best_area":      round(best_area, 4),
            "best_area_norm": round(best_area / (self.baseline_area + 1e-8), 6),
            "delay":          round(delay, 4),
            "n_selected":     n_selected,
            "q_loss":         round(metrics.get("q_loss", 0.0), 6),
            "epsilon":        round(metrics.get("epsilon", 0.0), 4),
            "mean_q":         round(metrics.get("mean_q", 0.0), 6),
            "elapsed_s":      round(elapsed, 1),
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

    def _rolling(self, values: List[float], w: int) -> List[float]:
        out = []
        for i in range(len(values)):
            s = max(0, i - w + 1)
            out.append(float(np.mean(values[s: i + 1])))
        return out

    def _save_plots(self):
        eps_axis = self.history["episode"]
        if not eps_axis:
            return

        # ── 1. reward ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        rewards = self.history["reward"]
        ax.plot(eps_axis, rewards, alpha=0.3, color="steelblue",
                linewidth=0.8, label="reward")
        ax.plot(eps_axis, self._rolling(rewards, _SMOOTH),
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
        area_norm = self.history["area_norm"]
        best_norm = self.history["best_area_norm"]
        ax.plot(eps_axis, area_norm, alpha=0.35, color="coral",
                linewidth=0.8, label="area (norm)")
        ax.plot(eps_axis, self._rolling(area_norm, _SMOOTH),
                color="coral", linewidth=1.8, label=f"rolling mean ({_SMOOTH} ep)")
        ax.plot(eps_axis, best_norm, color="darkred", linewidth=1.4,
                linestyle="--", label="best area so far")
        ax.axhline(1.0, color="gray", linewidth=0.8,
                   linestyle=":", label="baseline (1.0)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Area / baseline")
        ax.set_title("Normalised area (lower = better)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "area.png"), dpi=150)
        plt.close(fig)

        # ── 3. selection size ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        n_sel = self.history["n_selected"]
        ax.plot(eps_axis, n_sel, alpha=0.35, color="mediumpurple", linewidth=0.8)
        ax.plot(eps_axis, self._rolling(n_sel, _SMOOTH),
                color="mediumpurple", linewidth=1.8, label=f"rolling mean ({_SMOOTH} ep)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Gates selected")
        ax.set_title("Selected set size over training")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "n_selected.png"), dpi=150)
        plt.close(fig)

        # ── 4. DQN-specific: Q-loss, epsilon, mean-Q ─────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for ax, key, color, label in zip(
            axes,
            ["q_loss", "epsilon", "mean_q"],
            ["royalblue", "seagreen", "darkorange"],
            ["Q loss (Huber)", "Epsilon (ε-greedy)", "Mean Q-value"],
        ):
            vals = self.history[key]
            ax.plot(eps_axis, vals, alpha=0.35, color=color, linewidth=0.8)
            ax.plot(eps_axis, self._rolling(vals, _SMOOTH),
                    color=color, linewidth=1.8)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        axes[-1].set_xlabel("Episode")
        fig.suptitle("DQN diagnostics")
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "losses.png"), dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 8. TRAINING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    cfg = _load_config()
    train = cfg.get("train", {})

    p = argparse.ArgumentParser()
    p.add_argument("--lib",        default=train.get("library",   "nan45"))
    p.add_argument("--bench",      default=train.get("benchmark", "benchmarks/s838a.bench"))
    p.add_argument("--n-select",   type=int,   default=train.get("n_select",   50))
    p.add_argument("--max-steps",  type=int,   default=train.get("max_steps",  500))
    p.add_argument("--episodes",   type=int,   default=train.get("episodes",   300))
    p.add_argument("--lr",         type=float, default=train.get("lr",         1e-4))
    p.add_argument("--seed",       type=int,   default=train.get("seed",       42))
    p.add_argument("--buffer-size", type=int,  default=10_000,
                   help="Replay buffer capacity")
    p.add_argument("--batch-size",  type=int,  default=64)
    p.add_argument("--target-update-freq", type=int, default=200,
                   help="Hard target network update every N gradient steps")
    p.add_argument("--eps-start",  type=float, default=1.0)
    p.add_argument("--eps-end",    type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=5_000)
    return p.parse_args(), cfg


def _print_best_selection(best_area: float, best_sel: List[int],
                          mapper: TechMapper) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    if not best_sel:
        print("No selection recorded yet.")
        print(bar)
        return
    improvement = (mapper.baseline_area - best_area) / mapper.baseline_area * 100
    print(f"Best area : {best_area:.3f}  "
          f"(baseline {mapper.baseline_area:.3f}, {improvement:+.1f}%)")
    print(f"Selected gates ({len(best_sel)}):")
    for idx in best_sel:
        parts = mapper.mutable_gates[idx].split()
        name, area_val = parts[1], parts[2]
        print(f"  [{idx:3d}] {name:<30s} area={area_val}")
    print(bar)


def main():
    args, cfg = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    genlib_path = _resolve_library(cfg, args.lib)
    bench_path  = os.path.join(_ROOT, args.bench)
    gen_dir     = os.path.join(_ROOT, "gen_newlibs") + os.sep
    bench_stem  = os.path.splitext(os.path.basename(bench_path))[0]
    temp_blif   = os.path.join(_ROOT, "temp_blifs", f"{bench_stem}_dqn_temp.blif")

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
            f"--n-select {args.n_select} is too large.")

    raw_feats = extract_cell_features(mapper.mutable_gates)
    norm_feats, _, _ = normalize_features(raw_feats)
    circuit = parse_bench(bench_path)

    env = CircuitEnv(
        mapper, norm_feats, circuit,
        n_select=args.n_select, max_steps=args.max_steps,
    )

    agent = DQNTrainer(
        n_total=mapper.num_arms,
        lr=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"dqn_{args.lib}_{bench_stem}_{timestamp}"
    run_dir   = os.path.join(_ROOT, "logs", run_name)
    logger    = RunLogger(
        run_dir,
        baseline_area=mapper.baseline_area,
        baseline_delay=mapper.baseline_delay,
        cfg_dict=vars(args),
    )
    print(f"Run dir  : {run_dir}\n")

    best_area = float("inf")
    best_sel: List[int] = []
    t0 = time.time()

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_metrics: dict = {"q_loss": 0.0, "epsilon": agent.epsilon, "mean_q": 0.0}
            update_count = 0

            for _ in range(args.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _, info = env.step(action)
                agent.store(obs, action, reward, next_obs, done)

                ep_reward += reward
                obs = next_obs

                m = agent.update()
                if m is not None:
                    ep_metrics["q_loss"]  += m["q_loss"]
                    ep_metrics["epsilon"]  = m["epsilon"]
                    ep_metrics["mean_q"]  += m["mean_q"]
                    update_count += 1

                if info["area"] < best_area:
                    best_area = info["area"]
                    best_sel  = sorted(env._selected)

                if done:
                    break

            if update_count > 0:
                ep_metrics["q_loss"] /= update_count
                ep_metrics["mean_q"] /= update_count

            elapsed = time.time() - t0
            logger.log(ep, ep_reward, info["area"], info["delay"],
                       info["n_selected"], best_area, ep_metrics, elapsed)

            if ep % 10 == 0:
                print(
                    f"Ep {ep:4d}/{args.episodes}  "
                    f"reward {ep_reward:+.4f}  "
                    f"area {info['area']:.3f}  "
                    f"best {best_area:.3f}  "
                    f"n_sel {info['n_selected']:3d}  "
                    f"Q_loss {ep_metrics['q_loss']:.4f}  "
                    f"eps {ep_metrics['epsilon']:.3f}  "
                    f"[{elapsed:.0f}s]"
                )

    except KeyboardInterrupt:
        print("\n[interrupted]")

    finally:
        _print_best_selection(best_area, best_sel, mapper)
        logger.finalize()


if __name__ == "__main__":
    main()
