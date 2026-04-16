import numpy as np
import matplotlib.pyplot as plt
import time

filepath = 'experiment/data/q_history_20260416_200818.npy'
q_history = np.load(filepath)

arm_history = q_history[:, 4]
for i in arm_history[:100]:
    print(i)
