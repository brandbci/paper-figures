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
data_file = '240111T0948_pubsub_vary_multi*.csv'

def parse_test_id(test_id):
    if isinstance(test_id, str):
        n_nodes, n_chans = test_id.split('-')
        n_nodes = int(n_nodes[:-1])
        n_chans = int(n_chans[:-1])
    else:
        n_nodes = test_id
        n_chans = DEFAULT_N_CHANS
    return n_nodes, n_chans

def get_csv_key(csv_file):
    return os.path.basename(csv_file).split('_')[-1].split('.')[0]

def csvs_to_df_dict(csv_pattern):
    csv_files = glob(os.path.join(DATA_DIR, csv_pattern))
    return {get_csv_key(f): pd.read_csv(f) 
             for f in csv_files}

# Configure matplotlib
matplotlib.style.use('seaborn-colorblind')
matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['font.size'] = 10

# make seaborn color palette match matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns_palette = sns.color_palette(colors)

os.makedirs(PLOT_DIR, exist_ok=True)
# %%

node_dfs = csvs_to_df_dict(data_file)

min_sample_count = node_dfs['4n-256c'][['t_publisher', 't_sub0'
                                        ]].diff(axis=1).iloc[:, 1:].shape[0]

latencies = []
test_list = sorted(node_dfs.keys())
n_nodes_list = []
n_chans_list = []
for test_id in test_list:
    node_df = node_dfs[test_id]
    n_nodes, n_chans = parse_test_id(test_id)

    fields = ['t_publisher'] + [f't_sub{i :d}' for i in range(n_nodes - 1)]
    latency_df = node_df[fields].diff(axis=1).iloc[:, 1:]
    if FIXED_SAMPLE_COUNT:
        # take the first N samples from each node
        latency_df = latency_df.iloc[:np.ceil(min_sample_count /
                                              (n_nodes - 1)).astype(int), :]
    latency = latency_df.values.ravel()[:min_sample_count] / 1e3
    latencies.append(latency)

    n_nodes_list.append(n_nodes)

# %%
fig, axes = plt.subplots(nrows=4,
                         ncols=2,
                         figsize=(7.5, 7.5),
                         sharey=True,
                         gridspec_kw={'width_ratios': [4, 1]})

step = 10
for i_n, n_nodes in enumerate([2, 3, 4, 5]):
    for i_c, n_chans in enumerate([128, 256, 512, 1024]):
        test_id = f'{n_nodes}n-{n_chans}c'
        if test_id in test_list:
            idx = test_list.index(test_id)
            latency = latencies[idx]

            bins = np.arange(latency.max() + step, step=step)
            for i_col in range(2):
                axes[i_n, i_col].hist(latency,
                                      bins=bins,
                                      histtype='step',
                                      label=n_chans)
    axes[i_n, 0].set_xlabel('Latency ($\mu$s)')
    axes[i_n, 0].set_ylabel('Samples (log scale)')
    axes[i_n, 0].set_title(f'{n_nodes} nodes', loc='left')
    axes[i_n, 0].set_xlim(0, 440)
    axes[i_n, 1].set_xlim(7930, 8040)
    axes[i_n, 0].legend(fontsize=8, loc="upper right", ncol=2, frameon=False)
    axes[i_n, 1].spines.left.set_visible(False)
    axes[i_n, 1].tick_params(left=False)
    for i_col in range(2):
        axes[i_n, i_col].set_yscale('log')

    d = 1  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)],
                  markersize=12,
                  linestyle="none",
                  color='k',
                  mec='k',
                  mew=1,
                  clip_on=False)
    axes[i_n, 0].plot([1], [0],
                      transform=axes[i_n, 0].transAxes,
                      **kwargs)
    axes[i_n, 1].plot([0], [0],
                      transform=axes[i_n, 1].transAxes,
                      **kwargs)

plt.tight_layout()
fig.subplots_adjust(wspace=0.025)
plt.savefig(os.path.join(PLOT_DIR, 'figS3.pdf'))

# %%
