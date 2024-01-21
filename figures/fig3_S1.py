# %%
import os
from copy import deepcopy
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Circle

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
    'fig3d': 't11_230507_012_cursor_pos*.csv',
    'figS1b-c': '230517T1634_ole_graph_timing.csv',
    'figS1d': 't11_230507_003_cursor_pos*.csv',
}

# %%
# Load data
timing_dfs = {}
timing_files = {k: v for k, v in DATA_FILES.items() if 'timing' in v}
for fig_name, data_file in timing_files.items():
    data_path = os.path.join(DATA_DIR, data_file)
    with open(data_path, 'r') as f:
        timing_dfs[fig_name] = pd.read_csv(f)

os.makedirs(PLOT_DIR, exist_ok=True)

# %%
# configure matplotlib
matplotlib.style.use('seaborn-colorblind')
matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['font.size'] = 10

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
# Plot cursor position from T11 session
cursor_pos_files = {k: v for k, v in DATA_FILES.items() if 'cursor_pos' in v}

def get_csv_key(csv_file):
    return os.path.basename(csv_file).split('_')[-1].split('.')[0]

def csvs_to_df_dict(csv_pattern):
    csv_files = glob(os.path.join(DATA_DIR, csv_pattern))
    return {get_csv_key(f): pd.read_csv(f) 
             for f in csv_files}

kin_fields = ['cursorData_X', 'cursorData_Y']
for fig_name, file_pattern in cursor_pos_files.items():
    # load data
    data_dict = csvs_to_df_dict(file_pattern)

    # load dataframes
    kin_df = data_dict['kin'].set_index('clock_time')
    trial_info = data_dict['trials']

    # determine conditions
    conditions = trial_info['cond_id'].unique()
    conditions = conditions[conditions > 0]
    conditions.sort()

    # plot targets for each condition
    cmap = cm.get_cmap('tab10')
    patches = []
    fig, ax = plt.subplots(figsize=(2, 2))
    for ic, cond in enumerate(conditions):
        cond_mask = trial_info['cond_id'] == cond
        cond_info = trial_info[cond_mask].iloc[0, :]
        x, y = cond_info[['trial_info_target_X', 'trial_info_target_Y']]
        radius = cond_info['trial_info_target_radius']
        ax.add_patch(
            Circle((x, y), radius, edgecolor='k', facecolor=((0, 0, 0, 0))))

    # find outward trials for this condition
    mask = np.all((
        trial_info['cond_id'] > 0,
    ),
                axis=0)
    tinfo = trial_info[mask]

    # plot data for each trial
    for trial_id in tinfo.index:
        info = tinfo.loc[trial_id, :]
        start_time = info['start_time']
        end_time = info['end_time']
        cond_id = info['cond_id']
        cursor_pos = kin_df.loc[start_time:end_time, kin_fields].values
        ax.plot(cursor_pos[0, 0],
                cursor_pos[0, 1],
                'o',
                color=cmap(cond_id - 1),
                markersize=2)
        ax.plot(cursor_pos[:, 0],
                cursor_pos[:, 1],
                color=cmap(cond_id - 1),
                linewidth=1)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{fig_name}.pdf'))

# %%
