import numpy as np
import matplotlib.pyplot as plt
import time

filepath = r'/Users/williamsu/Documents/ntu/lecture/32/project/MapTune/experiment/data/random/cell_history_20260418_040339.npy'
cell_history = np.load(filepath)

for i in cell_history[:3]:
    print(len(i))
