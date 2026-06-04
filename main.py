"""Smoke test for GradMapper class."""

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 backport

from src.gradmap_mapper import GradMapper

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

bench = cfg["benchmarks"]["gradmap"][0]  # "adder"
mapper = GradMapper(bench, cfg, steps=50)

print(f"Benchmark : {mapper.bench_stem}")
print(f"Match file: {mapper._match_path}")
print()

delay, area = mapper.run(verbose=True)
reward = mapper.calculate_reward(delay, area)

print(f"\n{'='*50}")
print(f"Benchmark  : {mapper.bench_stem}")
print(f"Baseline   : delay={mapper.baseline_delay:.2f} ps  area={mapper.baseline_area:.4f}")
print(f"Best       : delay={delay:.2f} ps  area={area:.4f}")
print(f"Best ADP   : {mapper._best_adp:.4f}")
print(f"Reward     : {reward:.4f}")
print(f"Used cells : {sum(mapper.used_cells.values())} total")
for cell, cnt in mapper.used_cells.most_common(5):
    print(f"  {cell:<40s} {cnt:5d}")
