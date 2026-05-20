# plot the results from the log files
import os
import pandas as pd
import matplotlib.pyplot as plt

log_path = r'logs/20260520_190521_sa_7nm_c880/'

# Load the CSV log
csv_file = os.path.join(log_path, 'metrics.csv')
df = pd.read_csv(csv_file)

# Plot cost and best_cost over steps
plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['cost'], label='Cost', color='blue')
plt.plot(df['step'], df['best_cost'], label='Best Cost', color='orange')
plt.xlabel('Step')
plt.ylabel('Cost')
plt.title('Cost vs Step')
plt.legend()
plt.grid()
plt.savefig(os.path.join(log_path, 'cost_plot.png'))
plt.show()
