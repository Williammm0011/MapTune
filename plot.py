import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log_paths', type=str, nargs='+',
                    default=[r'logs/20260520_190521_sa_7nm_c880/'])
args = parser.parse_args()
log_paths = args.log_paths


# collect all metrics.csv under given paths
csv_files = []
for log_path in log_paths:
    csv_files.extend(glob.glob(os.path.join(
        log_path, '**/metrics.csv'), recursive=True))

plt.figure(figsize=(12, 6))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    label = os.path.basename(os.path.dirname(csv_file))

    if len(csv_files) == 1:
        plt.plot(df['step'], df['cost'], linestyle='--',
                 alpha=0.5, label=f'{label} Cost')
    plt.plot(df['step'], df['best_cost'], label=f'{label} Best Cost')

plt.xlabel('Step')
plt.ylabel('Cost')
plt.ylim(bottom=0.6, top=1.2)
plt.title('Cost vs Step')
plt.legend()
plt.grid()

out_dir = log_paths[0]
plt.savefig(os.path.join(out_dir, 'cost_plot.png'))
plt.show()
