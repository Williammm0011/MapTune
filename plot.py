import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('log_paths', type=str, nargs='+',
                    default=[r'logs/20260520_190521_sa_7nm_c880/'])
args = parser.parse_args()
log_paths = args.log_paths

plt.figure(figsize=(12, 6))

for log_path in log_paths:
    csv_file = os.path.join(log_path, 'metrics.csv')
    df = pd.read_csv(csv_file)
    label = os.path.basename(os.path.normpath(log_path))

    if len(log_paths) == 1:
        plt.plot(df['step'], df['cost'], linestyle='--',
                 alpha=0.5, label=f'{label} Cost')
    plt.plot(df['step'], df['best_cost'], label=f'{label} Best Cost')

plt.xlabel('Step')
plt.ylabel('Cost')
plt.ylim(0, 1.5)
plt.title('Cost vs Step')
plt.legend()
plt.grid()

out_dir = log_paths[0]
plt.savefig(os.path.join(out_dir, 'cost_plot.png'))
plt.show()
