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
# whether to use the same number of samples for each condition
FIXED_SAMPLE_COUNT = True
# number of samples to use for each condition
N_SAMPLES = 300_000
# whether to share the x-axis for histograms
HIST_SHAREX = True
# directory to save plots
PLOT_DIR = 'plots'
# data files
vary_channels_file = '220715T0846_pubsub_vary_channels.csv'
vary_sample_rate_file = '230522T2042_pubsub_vary_sample_rate.csv'
vary_nodes_file = '220714T1515_pubsub_vary_num_nodes*.csv'

# %%
# Set up plotting
# configure matplotlib
matplotlib.style.use('seaborn-colorblind')
matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['font.size'] = 10

# make seaborn color palette match matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns_palette = sns.color_palette(colors)

# boxplot arguments (for marking the median line)
bp_kwargs = dict(linewidth=0.2,
                 orient='h',
                 showfliers=False,
                 showbox=False,
                 showcaps=False,
                 whiskerprops={'visible': False},
                 width=0.9)

# make plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

# %%
# Load data
def get_csv_key(csv_file):
    return int(os.path.basename(csv_file).split('_')[-1].split('.')[0])

def csvs_to_df_dict(csv_pattern):
    csv_files = glob(os.path.join(DATA_DIR, csv_pattern))
    return {get_csv_key(f): pd.read_csv(f) 
             for f in csv_files}

ch_df = pd.read_csv(os.path.join(DATA_DIR, vary_channels_file))
sr_df = pd.read_csv(os.path.join(DATA_DIR, vary_sample_rate_file))
node_dfs = csvs_to_df_dict(vary_nodes_file)

# %%
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(7.5, 3.5), sharex=True)
# Plot 1: Vary channels
ip = 0

ch_df['latency'] = ch_df['t_sub'] - ch_df['t_pub']

n_channels_list = sorted(ch_df['n_channels'].unique())
latencies = []
for n_channels in n_channels_list:
    latency = ch_df['latency'][ch_df['n_channels'] == n_channels]
    latencies.append(latency.values[:N_SAMPLES] / 1e3)  # convert ns to us
latency_arr = np.array(latencies).T

yticks = np.arange(len(n_channels_list))
sns.violinplot(data=latencies,
               scale='width',
               linewidth=0.2,
               orient='h',
               palette=sns_palette,
               ax=axes[0, ip],
               markersize=2)
sns.boxplot(data=latencies, ax=axes[0, ip], **bp_kwargs)
axes[0, ip].set_ylabel('Channels')
axes[0, ip].set_xlabel('Latency ($\mu$s)')
axes[0, ip].set_yticks(ticks=yticks, labels=n_channels_list)
axes[0, ip].set_xticks(ticks=np.arange(0, 700, step=200))
axes[0, ip].xaxis.set_tick_params(labelbottom=True)

step = 10
bins = np.arange(np.ceil(latency_arr.max()) + step, step=step)
for n_channels, latency in zip(n_channels_list, latencies):
    axes[1, ip].hist(latency,
                     bins=bins,
                     histtype='step',
                     label=f'{n_channels}')
axes[1, ip].set_xlabel('Latency ($\mu$s)')
axes[1, ip].set_ylabel('Samples\n(log scale)')
axes[1, ip].set_yscale('log')
axes[1, ip].legend(fontsize=8,
                   bbox_to_anchor=(0, 1, 1, 0),
                   loc="lower left",
                   ncol=2,
                   frameon=False)

min_tick = 0
max_tick = np.floor(np.log10(axes[1, ip].get_ylim()[1])).astype(int)
ticks = 10**np.arange(min_tick, max_tick, step=2)
axes[1, ip].set_yticks(ticks)

# Plot 2: Vary sample rate
ip = 1

sr_df['latency'] = sr_df['t_sub'] - sr_df['t_pub']

sample_rate_list = sorted(sr_df['sample_rate'].unique())
latencies = []
for sample_rate in sample_rate_list:
    latency = sr_df['latency'][sr_df['sample_rate'] == sample_rate]
    latencies.append(latency.values / 1e3)  # convert ns to us
latency_arr = np.array(latencies).T

sns.violinplot(data=latencies,
               scale='width',
               linewidth=0.2,
               orient='h',
               palette=sns_palette,
               ax=axes[0, ip])
sns.boxplot(data=latencies, ax=axes[0, ip], **bp_kwargs)
axes[0, ip].set_ylabel('Rate (Hz)')
axes[0, ip].set_xlabel('Latency ($\mu$s)')
axes[0, ip].set_yticks(ticks=np.arange(len(sample_rate_list)),
                       labels=sample_rate_list)
axes[0, ip].xaxis.set_tick_params(labelbottom=True)

step = 10
bins = np.arange(np.ceil(latency_arr.max()) + step, step=step)
for sample_rate, latency in zip(sample_rate_list, latencies):
    axes[1, ip].hist(latency,
                     bins=bins,
                     histtype='step',
                     label=f'{sample_rate}')
axes[1, ip].set_xlabel('Latency ($\mu$s)')
axes[1, ip].set_yscale('log')
axes[1, ip].legend(fontsize=8,
                   bbox_to_anchor=(0, 1, 1, 0),
                   loc="lower left",
                   ncol=2,
                   frameon=False)
min_tick = 0
max_tick = np.floor(np.log10(axes[1, ip].get_ylim()[1])).astype(int)
ticks = 10**np.arange(min_tick, max_tick, step=2)
axes[1, ip].set_yticks(ticks)

# Plot 3: Vary nodes
ip = 2

min_sample_count = node_dfs[2][['t_publisher',
                                't_sub0']].diff(axis=1).iloc[:, 1:].shape[0]

latencies = []
n_nodes_list = sorted(node_dfs.keys())
for n_nodes in n_nodes_list:
    node_df = node_dfs[n_nodes]
    fields = ['t_publisher'] + [f't_sub{i :d}' for i in range(n_nodes - 1)]
    latency_df = node_df[fields].diff(axis=1).iloc[:, 1:]
    if FIXED_SAMPLE_COUNT:
        # take the first N samples from each node
        latency_df = latency_df.iloc[:np.ceil(min_sample_count /
                                              (n_nodes - 1)).astype(int), :]
    latency = latency_df.values.ravel()[:min_sample_count] / 1e3
    latencies.append(latency)

mean = np.array([np.mean(lat) for lat in latencies])
std = np.array([np.std(lat) for lat in latencies])
sns.violinplot(data=latencies,
               scale='width',
               linewidth=0.2,
               orient='h',
               palette=sns_palette,
               ax=axes[0, ip])
sns.boxplot(data=latencies, ax=axes[0, ip], palette=sns_palette, **bp_kwargs)
axes[0, ip].set_ylabel('Nodes')
axes[0, ip].set_xlabel('Latency ($\mu$s)')
axes[0, ip].set_yticks(ticks=np.arange(len(n_nodes_list)), labels=n_nodes_list)
axes[0, ip].xaxis.set_tick_params(labelbottom=True)

step = 10
for n_nodes, latency in zip(n_nodes_list, latencies):
    bins = np.arange(latency.max() + step, step=step)
    axes[1, ip].hist(latency, bins=bins, histtype='step', label=f'{n_nodes}')
axes[1, ip].set_xlabel('Latency ($\mu$s)')
axes[1, ip].set_yscale('log')
axes[1, ip].legend(fontsize=8,
                   bbox_to_anchor=(0, 1, 1, 0),
                   loc="lower left",
                   ncol=2,
                   frameon=False)

min_tick = 0
max_tick = np.floor(np.log10(axes[1, ip].get_ylim()[1])).astype(int)
ticks = 10**np.arange(min_tick, max_tick, step=2)
axes[1, ip].set_yticks(ticks)

# share x axis
if HIST_SHAREX:
    ncols = axes.shape[1]
    xlims = [axes[1, ip].get_xlim()[1] for ip in range(ncols)]
    for ip in range(ncols):
        axes[1, ip].set_xlim(0, max(xlims))

# start x axis at zero in the top row
ncols = axes.shape[1]
xlims = [axes[0, ip].get_xlim()[1] for ip in range(ncols)]
for ip in range(ncols):
    axes[0, ip].set_xlim(0, max(xlims))

# format figure and save
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig2.pdf'), dpi=300)

# %%
