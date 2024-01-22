# %%
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# directory containing the data files
DATA_DIR = '/snel/share/share/data/brand/lvm'
# directory to save plots
PLOT_DIR = 'plots'
# data files to load
NDT_DATA_FILE = '221122T0711_ndt_latency.csv'
LFADS_DATA_FILE = '220824T1546_lfads_latency.csv'
# number of samples to use
N_SAMPLES = 30_000

# %%
# Configure plotting
matplotlib.style.use('seaborn-colorblind')
matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['font.size'] = 10

# make seaborn color palette match matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns_palette = sns.color_palette(colors)

os.makedirs(PLOT_DIR, exist_ok=True)

# %%
# Load data
with open(os.path.join(DATA_DIR, NDT_DATA_FILE), 'rb') as f:
    ndt_latency_df = pd.read_csv(f)

with open(os.path.join(DATA_DIR, LFADS_DATA_FILE), 'rb') as f:
    lfads_latency_df = pd.read_csv(f)

# %%
# Make plots
fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(3.5, 2.333))
# plot per-node latency as a histogram
step = 10e-3
ax = axes

latency = ndt_latency_df['ts_ndt'].values * 1e-6
bins = np.arange(latency.max() + step, step=step)
ax.hist(latency, bins=bins, histtype='step', label='NDT')

latency = lfads_latency_df['ts_lfads'].values * 1e-6
bins = np.arange(latency.max() + step, step=step)
ax.hist(latency, bins=bins, histtype='step', label='LFADS')

ax.set_xlabel('Node Latency (ms)')
ax.set_ylabel('Samples (log scale)')
ax.set_yscale('log')

min_tick = 0
max_tick = np.floor(np.log10(ax.get_ylim()[1])).astype(int)
ticks = np.logspace(min_tick, max_tick, max_tick - min_tick + 1)
ax.set_yticks(ticks)
ax.set_xlim(0, np.ceil(ax.get_xlim()[1]))

ax.legend(fontsize=8, frameon=False, loc='upper center')

# format figure and save
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig4c.pdf'), dpi=300)
# %%
