import matplotlib.pyplot as plt
import random
import math
import sys
import os
import numpy as np
import subprocess
from subprocess import PIPE
import re
import time

# ============== Configuration ==============
NUM_ITERATION = 100000
REWARD_COEFFICIENT = 0.2
REWARD_EXPONENTIAL_DECAY = 500
EXPLORATION_PARAMETER = 50
STOP_NO_PROGRESS_THRESHOLD = 500
# =========================================

genlib_origin = sys.argv[-1]
lib_origin = genlib_origin[:-7] + '.lib'
design = sys.argv[-2]
sample_gate = int(sys.argv[-3])
temp_blif = "temp_blifs/" + design[:-5] + "_ucb_temp.blif"
lib_path = "gen_newlibs/"

start = time.time()
abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (
    genlib_origin, design, temp_blif, lib_origin, temp_blif)
res = subprocess.check_output(('abc', '-c', abc_cmd))
match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
# Baseline
max_delay = float(match_d.group(1))
max_area = float(match_a.group(1))

print("Baseline Delay:", max_delay)
print("Baseline Area:", max_area)

# Mapper call


def technology_mapper(genlib_origin, partial_cell_library):
    with open(genlib_origin, 'r') as f:
        # f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE AND2") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv")
                   and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    with open(genlib_origin, 'r') as f:
        # f_keep = [line.strip() for line in f if any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
        f_keep = [line.strip() for line in f if line.startswith("GATE AND2") or line.startswith("GATE BUF") or line.startswith("GATE INV") or line.startswith("GATE sky130_fd_sc_hd__buf") or line.startswith("GATE sky130_fd_sc_hd__inv") or line.startswith(
            "GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") or line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
    f.close()
    lines_partial = [f_lines[i] for i in partial_cell_library]
    lines_partial = lines_partial + f_keep
    output_genlib_file = lib_path + design + "_" + \
        str(len(lines_partial)) + "_ucb_samplelib.genlib"
    with open(output_genlib_file, 'w') as out_gen:
        for line in lines_partial:
            out_gen.write(line + '\n')
    out_gen.close()

    abc_cmd = "read %s;read %s; map; write %s; read %s;read -m %s; ps; topo; upsize; dnsize; stime; " % (
        output_genlib_file, design, temp_blif, lib_origin, temp_blif)
    res = subprocess.check_output(('abc', '-c', abc_cmd))
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float("NaN"), float("NaN")
    return delay, area

# Reward calculation


# def calculate_reward(max_delay, max_area, delay, area):
def calculate_reward(avg_delay, avg_area, delay, area):
    normalized_delay = delay / avg_delay
    normalized_area = area / avg_area
    # sqrt -> log
    # return -np.log(normalized_delay * normalized_area)

    # log -> sigmoid
    return 0.5 - 1 / (1 + np.exp(5 * (1 - normalized_delay * normalized_area)))


# UCB MAB Class


class UCB_MAB:
    def __init__(self, num_arms, c, sample_gate):
        self.num_arms = num_arms
        self.c = c  # Exploration parameter for UCB
        self.q_values = [0.0] * num_arms
        self.counts = [0] * num_arms
        self.sample_gate = sample_gate

    def select_action(self):
        selected_cells = set()

        # Exploration (ensure each arm is tried at least once)
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                selected_cells.add(arm)
                if len(selected_cells) == self.sample_gate:
                    break

        # Exploitation (choose the remaining based on UCB)
        remaining_cells = [arm for arm in range(
            self.num_arms) if arm not in selected_cells]
        total_counts = sum(self.counts)
        # print(total_counts)
        ucb_values = [0.0] * self.num_arms
        for arm in remaining_cells:
            if self.counts[arm] > 0:
                average_reward = self.q_values[arm]
                ucb_values[arm] = average_reward + self.c * \
                    math.sqrt(math.log(total_counts) / self.counts[arm])
        while len(selected_cells) < self.sample_gate:
            if all(math.isinf(val) or math.isnan(val) for val in ucb_values):
                selected_cell = random.choice(remaining_cells)
            else:
                # Use softmax to convert ucb_values to probabilities
                x = np.array(ucb_values)
                x -= x.max()
                probs = np.exp(x) / np.exp(x).sum()
                selected_cell = np.random.choice(len(x), p=probs)
                # selected_cell = ucb_values.index(max(ucb_values))
            if selected_cell not in selected_cells:
                selected_cells.add(selected_cell)
                ucb_values[selected_cell] = float('-inf')

        return list(selected_cells)

    def update(self, selected_arm, reward):
        for arm in selected_arm:
            self.counts[arm] += 1
            # self.q_values[arm] = (self.q_values[arm] *
            #                       self.counts[arm] + reward) / self.counts[arm]
            # Exponential moving average
            self.q_values[arm] += reward * \
                REWARD_COEFFICIENT * \
                np.exp(-self.counts[arm] / REWARD_EXPONENTIAL_DECAY)


# Initialization
num_cells_select = sample_gate
with open(genlib_origin, 'r') as f:
    # f_lines = [line.strip() for line in f if line.startswith("GATE") and not any(substr in line for substr in ["BUF", "INV", "inv", "buf"])]
    f_lines = [line.strip() for line in f if line.startswith("GATE") and not line.startswith("GATE AND2") and not line.startswith("GATE BUF") and not line.startswith("GATE INV") and not line.startswith("GATE sky130_fd_sc_hd__buf") and not line.startswith("GATE sky130_fd_sc_hd__inv")
               and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__buf") and not line.startswith("GATE gf180mcu_fd_sc_mcu7t5v0__inv")]
f.close()
num_arms = len(f_lines)
mab = UCB_MAB(num_arms, c=EXPLORATION_PARAMETER, sample_gate=num_cells_select)
best_cells = None
best_result = (float('inf'), float('inf'))
best_reward = -float('inf')

# =================== My Experiment ===================
# To store Q-values, delay, area history for all iterations
history = [[], [], []]
cell_history = []

# Main Loop


for i in range(NUM_ITERATION):
    # print("Iteration: ", i, end='\r')

    selected_cells = random.sample(range(num_arms), num_cells_select)
    try:
        delay, area = technology_mapper(genlib_origin, selected_cells)
        if delay == float("NaN") or area == float("NaN"):
            reward = -float('inf')
        else:
            avg_delay = np.mean(history[1][-20:]) if history[1] else max_delay
            avg_area = np.mean(history[2][-20:]) if history[2] else max_area
            reward = calculate_reward(avg_delay, avg_area, delay, area)
            # fix the wdith of the iteration number to 2 and the reward to 4 decimal places
            print(
                f"Iteration: {i:3}, Product: {delay * area:10.4f}, Reward: {reward:10.4f}", end='\n')
    except Exception:
        reward = -float('inf')

    if delay * area < best_result[0] * best_result[1]:
        # if reward > best_reward:
        no_progess_count = 0
        print("\nIteration: ", i)
        best_reward = reward
        print(f"Current best reward: {best_reward:.4f}")
        best_result = (delay, area)
        print(
            f"Current best result: {(delay * area)/(max_delay * max_area)*100:.2f}%")
        best_cells = selected_cells
    mab.update(selected_cells, reward)

    # record q values for all arms
    history[0].append(mab.q_values.copy())
    history[1].append(delay)
    history[2].append(area)
    cell_history.append(selected_cells)


end = time.time()
runtime = end-start

print("Best Cells:", best_cells)
print("Best Delay:", best_result[0])
print("Best Area:", best_result[1])
print("Best Reward:", best_reward)
print("Total time:", runtime)


# =============== Save q_history to a file ===============
q_history = np.array(history[0])
delay_history = np.array(history[1])
area_history = np.array(history[2])
cell_history = np.array(cell_history)
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Optionally, save delay and area history as well
np.save(
    f"experiment/data/random/product_history_{timestamp}.npy", delay_history*area_history)
np.save(
    f"experiment/data/random/cell_history_{timestamp}.npy", cell_history)

product = np.array(delay_history * area_history)
product.sort()

# plot area * delay history
plt.figure(figsize=(8, 6))
plt.plot(product, label='Delay * Area')
# draw the baseline as a horizontal line
plt.axhline(y=max_delay * max_area, color='r',
            linestyle='--', label='Baseline Delay * Area')
plt.xlabel('Iteration')
plt.ylabel('Delay * Area')
plt.title(
    f'Delay * Area History {int(best_result[0] * best_result[1] / (max_delay * max_area) * 100)}% of Baseline')
plt.grid()
plt.legend()
plt.savefig(
    f"experiment/plots/random/delay_area_product_history_{timestamp}.png")
plt.show()
plt.close()

'''
# plot delay and area history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# draw the baseline as a horizontal line
plt.axhline(y=max_delay, color='r', linestyle='--', label='Baseline Delay')
plt.plot(delay_history, label='Delay')
plt.xlabel('Iteration')
plt.ylabel('Delay (ps)')
plt.title('Delay History')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
# draw the baseline as a horizontal line
plt.axhline(y=max_area, color='r', linestyle='--', label='Baseline Area')
plt.plot(area_history, label='Area')
plt.xlabel('Iteration')
plt.ylabel('Area')
plt.title('Area History')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f"experiment/plots/delay_area_history_{timestamp}.png")
plt.show()
plt.close()
'''
