import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log_paths', type=str, nargs='+', default=['logs/'])
args = parser.parse_args()
log_paths = args.log_paths

for log_path in log_paths:
    subfolders = sorted(glob.glob(os.path.join(log_path, '*/')))

    groups = {}
    for subfolder in subfolders:
        name = os.path.basename(os.path.normpath(subfolder))
        m = re.match(r'^\d{8}_\d{6}_(sa|random)_multi_([^_]+)_(.+)$', name)
        if not m:
            continue

        method, lib, benchmark = m.group(1), m.group(2), m.group(3)
        key = (lib, benchmark)
        groups.setdefault(key, {})[method] = subfolder

    for (lib, benchmark), methods in groups.items():
        plt.figure(figsize=(12, 6))

        for method, subfolder in methods.items():
            color = 'red' if method == 'sa' else 'blue'
            agent_csvs = sorted(glob.glob(os.path.join(
                subfolder, '**/metrics.csv'), recursive=True))
            for csv_file in agent_csvs:
                df = pd.read_csv(csv_file)
                agent_label = os.path.basename(os.path.dirname(csv_file))
                improvement = 1 - df['best_cost']
                plt.plot(df['step'], improvement, color=color,
                         alpha=0.7, label=f'{method} {agent_label}')

        plt.xlabel('Step')
        plt.ylabel('Improvement (1 - Best Cost)')
        plt.ylim(-0.2, 0.6)
        plt.title(f'{lib} — {benchmark}')
        plt.legend()
        plt.grid()

        out_path = os.path.join(log_path, f'{lib}_{benchmark}.png')
        plt.savefig(out_path)
        plt.close()
        print(f'Saved: {out_path}')
