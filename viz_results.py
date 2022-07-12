import matplotlib.pyplot as plt
import numpy as np
import time

envs2=(
	"halfcheetah-random-v2",
	"hopper-random-v2",
	"walker2d-random-v2",
	"halfcheetah-medium-v2",
	"hopper-medium-v2",
	"walker2d-medium-v2",
	"halfcheetah-medium-replay-v2",
	"hopper-medium-replay-v2",
	"walker2d-medium-replay-v2",
	"halfcheetah-medium-expert-v2",
	"hopper-medium-expert-v2",
	"walker2d-medium-expert-v2",
	)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# file_name = 'expriments/exp_td3_wbc/results/TD3_BC_hopper-medium-replay-v2_10.npy'
# log = np.load(file_name, allow_pickle=True)
# plt.plot(moving_average(log, 30))
# plt.title(file_name)
# plt.show()
# file_name2 = 'expriments/exp_td3_wbc/infos/TD3_BC_hopper-medium-replay-v2_10.npy'
# log2 = np.load(file_name2, allow_pickle=True)
# bc_ratio = []
# for i in log2:
#     bc_ratio.append(i['Q'].item())
# plt.plot(moving_average(bc_ratio, 1))
# plt.title(file_name2)
# plt.show()

import os
dir = 'results'
seeds = ['0', '1', '10','11','12']
all_files = os.listdir(dir)
all_files.sort(key=lambda fn: os.path.getmtime(dir+'/'+fn))
files = []
for i in all_files:
    if i.split('.')[-1] == 'npy' and i.split('.')[0].split('_')[-1] in seeds:
        files.append(i)

files_sorted = []
for seed in seeds:
    for i in envs2:
        name = 'TD3_wBC_' + i + '_' + seed + '.npy'
        assert name in files
        files_sorted.append(name)
files = files_sorted[:]

scores = []
for i in range(len(files)):
    name = dir + files[i]
    log = np.load(name, allow_pickle=True)
    ret = np.mean(log[-1:])
    scores.append(ret)
sum_score = np.sum(scores) / len(seeds)