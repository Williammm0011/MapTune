import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

filepath = 'experiment/data/q_history_20260416_200818.npy'
q_history = np.load(filepath)

fig, ax = plt.subplots(figsize=(12, 6))
sc = ax.scatter([], [], s=10)

ax.set_xlim(0, q_history.shape[1] - 1)
ax.set_ylim(-10, 0)  # Adjust based on expected Q value range
ax.set_xlabel('Arm')
ax.set_ylabel('Q Value')
ax.grid(True)

title = ax.set_title('')


def update(frame):
    sc.set_offsets(np.column_stack([
        range(q_history.shape[1]),
        q_history[frame]
    ]))
    title.set_text(f'Q Values — Iteration {frame}')
    return sc, title


ani = animation.FuncAnimation(
    fig, update,
    frames=q_history.shape[0],
    interval=100,      # ms per frame
    blit=True
)

# add timestamp to filename
timestamp = time.strftime("%Y%m%d_%H%M%S")
ani.save(r'/Users/williamsu/Documents/ntu/lecture/32/project/MapTune/experiment/data/q_history_' + timestamp + '.gif',
         writer='pillow', fps=10)
plt.close()
print("Saved q_history_" + timestamp + ".gif")
