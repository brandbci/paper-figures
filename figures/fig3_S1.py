# %%
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# directory containing the data files
DATA_DIR = '/snel/share/share/data/brand/bci_control_t11'
# directory to save plots
PLOT_DIR = 'plots'
# whether the decoder used the normalized or unnormalized data
USE_NORM = True
# whether to use the maximum latency when there are multiple streams for the
# same node type
USE_MAX_LATENCY = True
# maximum x-axis value
FIXED_XMAX = None
# data files to load
DATA_FILES = {
    'fig3b-c': '230517T1731_rnn_graph_timing.csv',
    'figS1b-c': '230517T1634_ole_graph_timing.csv'
}

# %%
# Load data
timing_dfs = {}
for fig_name, data_file in DATA_FILES.items():
    data_path = os.path.join(DATA_DIR, data_file)
    with open(data_path, 'r') as f:
        timing_dfs[fig_name] = pd.read_csv(f)

os.makedirs(PLOT_DIR, exist_ok=True)

matplotlib.style.use('seaborn-colorblind')
matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['font.size'] = 10

# %%
# make matplotlib color palette match seaborn
sns.set_palette("colorblind")

# %%
ts_labels = {
    'ts_nn1': 'NSP Input',
    'ts_tc1': 'Thresholding',
    'ts_bs': 'Binning',
    'ts_nr': 'Normalization',
    'ts_wf': 'Decoder',
    'ts_cv': 'Smoothing',
    'ts_cd': 'Task Logic'
}
ts_labels['ts_nn2'] = ts_labels['ts_nn1']
ts_labels['ts_tc2'] = ts_labels['ts_tc1']
ts_labels['ts_bs1'] = ts_labels['ts_bs2'] = 'Binning'

colors = {
    'ts_nn1': 'C0',
    'ts_tc1': 'C0',
    'ts_bs': 'C1',
    'ts_lfads': 'C9',
    'ts_nr': 'C7',
    'ts_wf': 'C3',
    'ts_cv': 'C4',
    'ts_cd': 'C2'
}
colors['ts_nn2'] = colors['ts_nn1']
colors['ts_tc2'] = 'C6'
colors['ts_bs1'] = colors['ts_bs']
colors['ts_bs2'] = 'C7'

# %%
# Make plots
for fig_name, timing_df in timing_dfs.items():
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(7.5, 2.5))
    axr = axes
    i = 1
    all_ts_fields = {}
    for i in range(2):
        ts_fields = [
            f'ts_nn{i + 1}', # 
            f'ts_tc{i + 1}', 'ts_bs', 'ts_nr', 'ts_wf', 'ts_cv',
            'ts_cd'
        ]
        if not USE_NORM:
            ts_fields.remove('ts_nr')
        all_ts_fields[i] = deepcopy(ts_fields)

    latency_df1 = timing_df[all_ts_fields[0]].diff(axis=1).iloc[:, 1:]
    latency_df2 = timing_df[all_ts_fields[1]].diff(axis=1).iloc[:, 1:]
    latency_df2.rename({'ts_bs': 'ts_bs2'}, axis=1, inplace=True)
    latency_df = latency_df1.join(latency_df2[['ts_bs2']])
    latency_df.rename({'ts_bs': 'ts_bs1'}, axis=1, inplace=True)

    max_latencies = np.max(np.stack((latency_df1, latency_df2)), axis=0)
    max_latency_df = pd.DataFrame(max_latencies, columns=latency_df1.columns)

    cs_latency_df1 = latency_df1.cumsum(axis=1)
    cs_latency_df2 = latency_df2.cumsum(axis=1)
    max_cs_latencies = np.max(np.stack((cs_latency_df1, cs_latency_df2)), axis=0)
    max_cs_latency_df = pd.DataFrame(max_cs_latencies,
                                    columns=cs_latency_df1.columns)

    # plot per-node latency as a histogram
    step = 10e-3
    plt_df = max_latency_df if USE_MAX_LATENCY else latency_df
    for field in plt_df.columns:
        label = ts_labels[field]
        latency = plt_df[field].values * 1e-6
        bins = np.arange(latency.max() + step, step=step)
        if not USE_MAX_LATENCY:
            if field.endswith('1'):
                label += ' 1'
            elif field.endswith('2'):
                label += ' 2'
        axr[0].hist(latency,
                    bins=bins,
                    color=colors[field],
                    histtype='step',
                    label=label)

    axr[0].set_xlabel('Node Latency (ms)')
    axr[0].set_ylabel('Samples')
    axr[0].set_yscale('log')
    axr[0].legend(fontsize=8, ncol=1, frameon=False)

    # plot cumulative latency as a horizontal violin plot
    labels = [ts_labels[field] for field in max_cs_latency_df.columns]
    palette = [colors[field] for field in max_cs_latency_df.columns]
    sns.violinplot(data=max_cs_latency_df * 1e-6,
                scale='width',
                linewidth=0.2,
                orient='h',
                ax=axr[1],
                palette=palette)
    axr[1].set_xlabel('Cumulative Latency (ms)')
    axr[1].set_yticks(ticks=np.arange(max_cs_latency_df.shape[1]))
    axr[1].set_yticklabels(labels=labels)

    axr[1].set_title(f'N = {timing_df.shape[0] :,}', loc='left')

    left, right = axr[1].get_xlim()
    bottom, top = axr[1].get_ylim()

    # # make the x-axes match
    ncols = len(axes.flat)
    xlims = [ax.get_xlim()[1] for ax in axes.flat]
    for ax in axes.flat:
        if FIXED_XMAX:
            ax.set_xlim(0, FIXED_XMAX)
        else:
            ax.set_xlim(0, max(xlims))

    # format figure and save
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{fig_name}.pdf'), dpi=300)
    plt.show()

# %%
