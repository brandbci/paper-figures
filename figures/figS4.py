# %%
import os
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# directory containing the data files
DATA_DIR = '/snel/share/share/data/brand/pubsub'
# directory to save plots
PLOT_DIR = 'plots'
# number of samples to use for each condition
N_SAMPLES = 300_000
# default number of channels to use when only varying nodes
DEFAULT_N_CHANS = 128
# whether to share the x-axis for histograms
HIST_SHAREX = True
# whether to use the same number of samples for each condition
FIXED_SAMPLE_COUNT = True
# data files
scaling_file = '240111T1501_pubsub_scaling*.csv'

# Configure matplotlib
matplotlib.style.use('seaborn-colorblind')
matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['font.size'] = 10

# make seaborn color palette match matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns_palette = sns.color_palette(colors)

# Helper functions
def parse_test_id(test_id):
    n_nodes, n_chans = test_id.split('-')
    n_nodes = int(n_nodes[:-1])
    n_chans = int(n_chans[:-1])
    return n_nodes, n_chans

def get_csv_key(csv_file):
    return os.path.basename(csv_file).split('_')[-1].split('.')[0]

def csvs_to_df_dict(csv_pattern):
    csv_files = glob(os.path.join(DATA_DIR, csv_pattern))
    return {get_csv_key(f): pd.read_csv(f) 
             for f in csv_files}

# %%
# Load data
node_dfs = csvs_to_df_dict(scaling_file)

# %%
# Plot latency distributions
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 3), sharex=True)

min_sample_count = node_dfs['4n-256c'][['t_publisher', 't_sub0']].diff(
    axis=1).iloc[:, 1:].shape[0] * 3

latencies = []
n_nodes_list = []
test_id_list = sorted(node_dfs.keys(), key=lambda x: parse_test_id(x)[0])
for test_id in test_id_list:
    n_nodes, n_chans = parse_test_id(test_id)
    node_df = node_dfs[test_id]
    fields = ['t_publisher'] + [f't_sub{i :d}' for i in range(n_nodes - 1)]
    latency_df = node_df[fields].diff(axis=1).iloc[:, 1:]
    if FIXED_SAMPLE_COUNT:
        # take the first N samples from each node
        latency_df = latency_df.iloc[:np.ceil(min_sample_count /
                                              (n_nodes - 1)).astype(int), :]
    latency = latency_df.values.ravel()[:min_sample_count] / 1e3
    latencies.append(latency)
    n_nodes_list.append(n_nodes)

mean = np.array([np.mean(lat) for lat in latencies])
std = np.array([np.std(lat) for lat in latencies])
sns.violinplot(data=latencies,
               scale='width',
               linewidth=0.2,
               palette=sns_palette,
               orient='h',
               ax=axes[0])
sns.boxplot(data=latencies,
            ax=axes[0],
            linewidth=0.4,
            orient='h',
            showfliers=False,
            showbox=False,
            showcaps=False,
            whiskerprops={'visible': False},
            width=0.9)
axes[0].set_ylabel('Nodes')
axes[0].set_yticks(ticks=np.arange(len(n_nodes_list)), labels=n_nodes_list)

step = 10
for n_nodes, latency in zip(n_nodes_list, latencies):
    bins = np.arange(latency.max() + step, step=step)
    axes[1].hist(latency, bins=bins, histtype='step', label=f'{n_nodes}')
axes[1].set_xlabel('Latency ($\mu$s)')
axes[1].set_yscale('log')
axes[1].legend(fontsize=8, loc="upper right", ncol=2, frameon=False)

min_tick = 0
max_tick = np.floor(np.log10(axes[1].get_ylim()[1])).astype(int)
ticks = 10**np.arange(min_tick, max_tick, step=2)
axes[1].set_yticks(ticks)

# start x axis at zero in the violin plot
axes[0].set_xlim(0, 100 * np.ceil(axes[0].get_xlim()[1] / 100))

# set labels
axes[0].set_xlabel('Latency ($\mu$s)')
axes[1].set_ylabel('Samples (log scale)')

# format figure and save
plt.tight_layout()
file_id = scaling_file.split('_')[0]
plt.savefig(os.path.join(PLOT_DIR, 'figS4.pdf'), dpi=300)
# %%
