# plot the reward curve in /Users/williamsu/Documents/ntu/lecture/32/project/MapTune/logs/7nm_s838a_20260514_202244/training.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(
    '/Users/williamsu/Documents/ntu/lecture/32/project/MapTune/logs/7nm_s838a_20260514_202244/training.csv')
plt.plot(df['reward'], label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Curve')
plt.legend()
plt.show()
