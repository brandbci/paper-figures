# %%
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns

DATA_DIR = '/snel/share/share/data/brand'
PLOT_DIR = 'plots'

# data files
speech_sim_data_file = 'mic_sim_data.mat'
speech_sim_latency_file = 'mic_sim_latencies.mat'
cursor_sim_data_file = 'sim_dataset.pkl'
cursor_sim_latency_file = '230808T0219_cursor_sim_latency.csv'

# configure matplotlib
matplotlib.style.use('seaborn-colorblind')
matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['font.size'] = 8

# make seaborn color palette match matplotlib
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns_palette = sns.color_palette(colors)

os.makedirs(PLOT_DIR, exist_ok=True)

# %%
# Load speech sim data
mic_sim_data = scipy.io.loadmat(
    os.path.join(DATA_DIR, 'speech_sim', speech_sim_data_file))

# Load cursor sim data
with open(os.path.join(DATA_DIR, 'simulator', cursor_sim_data_file),
          'rb') as f:
    gdf = pickle.load(f)

gdf['ts_end_fr'][:-5] = gdf['ts_end_fr'][5:]
gdf['ts_end_thres'][:-5] = gdf['ts_end_thres'][5:]

# Calculate position
gdf['mouse_pos_x'] = np.cumsum(gdf['mouse_vel_x'])
gdf['mouse_pos_y'] = np.cumsum(gdf['mouse_vel_y'])

gdf.dropna(inplace=True)

# %%
fig_height = 1.5
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(7.5, fig_height))
axc = axes

thresholds = np.stack(gdf['thresholds1'])

T0 = 20000
TF = T0 + 1541
N_samples = 30 * 50  # how many samples to put in voltage plot
n_channels = 96
n_30k_channels = 6

ax = axc[0]
ax.set_title('Cursor Velocity')
t = gdf['i_thres'][T0:TF] - T0
for i, dim in enumerate(['x', 'y']):
    ax.plot(t, gdf[f'mouse_vel_{dim}'][T0:TF], label=f'{dim}')
ax.set_xlim([np.min(t), np.max(t)])
ax.set_ylabel(f'Velocity (a.u.)')
ax.legend()
ax.set_xlabel('Time (ms)')

ax = axc[1]
ax.set_title('Firing Rates')
fr = np.stack(gdf['rates'][T0:TF]).T[:n_channels, :]
ax.imshow(fr, aspect='auto', interpolation=None)
ax.set_ylabel('Channels')
ax.set_xlabel('Time (ms)')
ax.set_yticks([])
ax.spines['left'].set_visible(False)

ax = axc[2]
ax.set_title('Spikes')
raster = 1 - thresholds[T0:TF].T[:n_channels, :]
n_y, n_x = raster.shape
im = ax.imshow(raster,
               aspect='auto',
               interpolation='none',
               vmin=0,
               vmax=1,
               cmap='gray')
ax.set_ylabel('Channels')
ax.set_xlabel('Time (ms)')
ax.set_yticks([])
ax.spines['left'].set_visible(False)

# Plot continuous data
if len(axc) > 3:
    ax = axc[3]
    n_neurons = 96
    scale_continuous1 = 600

    continuous1 = np.vstack(gdf['continuous1']).T

    N_arrays = 1
    tslice = slice(T0 * 30, (T0 * 30) + N_samples)
    N_per_array = int(n_neurons / N_arrays)
    for r in range(0, int(N_arrays)):
        ax.set_title(f'Array {r+1}')
        for i in range(n_channels - n_30k_channels, n_channels):
            data_30k = continuous1[int(r * N_per_array + i),
                                   tslice] + i * scale_continuous1
            ax.plot(np.linspace(0, len(data_30k) / 30, num=len(data_30k)),
                    data_30k,
                    label=f'Channel {int(r*N_per_array+i)}')
    ax.set_title(f'Voltages')
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Channels')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig5c.pdf'))

# %%
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(7.5, fig_height))
axs = axes
# audio waveform
ax = axs[0]
mic_time = (1e3 * np.array(range(0, len(mic_sim_data['microphone_data'][0]))) /
            1100)
ax.plot(mic_time, mic_sim_data['microphone_data'][0])
ax.set_xlim([0, np.max(mic_time)])
ax.set_xlabel('Time (ms)')
ax.set_title('Audio')
ax.set_yticks([])
ax.set_ylabel('Magnitude (a.u.)')

# MFCC
ax = axs[1]
n_y, n_x = mic_sim_data['mfcc_data'].shape
im = ax.imshow(mic_sim_data['mfcc_data'],
               aspect='auto',
               interpolation='none',
               clim=(-3, 3),
               cmap='plasma',
               extent=(0, n_x * 5, n_y, 0))
ax.set_title('MFCC')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Coefficients')
ax.set_yticks([])
ax.spines['left'].set_visible(False)

# Firing rate
ax = axs[2]
n_y, n_x = mic_sim_data['firing_rates'].shape
im = ax.imshow(mic_sim_data['firing_rates'],
               aspect='auto',
               interpolation='none',
               cmap='viridis',
               extent=(0, n_x * 5, n_y, 0))
ax.set_title('Firing Rates')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Channels')
ax.set_yticks([])
ax.spines['left'].set_visible(False)

# Spikes
ax = axs[3]
n_y, n_x = mic_sim_data['simulated_spikes'].shape
im = ax.imshow(mic_sim_data['simulated_spikes'],
               aspect='auto',
               interpolation='none',
               vmin=0,
               vmax=1,
               cmap='gray',
               extent=(0, n_x, n_y, 0))
ax.set_title('Spikes')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Channels')
ax.set_yticks([])
ax.spines['left'].set_visible(False)

# Plot continuous data
if len(axs) > 3:
    ax = axs[4]

    T0 = 0
    N_samples = 30 * 50  # how many samples to put in voltage plot

    n_neurons = 96
    scale_continuous1 = 600

    continuous1 = mic_sim_data['rawNeural']

    N_arrays = 1
    tslice = slice(T0 * 30, (T0 * 30) + N_samples)
    N_per_array = int(n_neurons / N_arrays)
    for r in range(0, int(N_arrays)):
        ax.set_title(f'Array {r+1}')
        for i in range(n_channels - n_30k_channels, n_channels):
            data_30k = continuous1[int(r * N_per_array + i),
                                   tslice] + i * scale_continuous1
            ax.plot(np.linspace(0, len(data_30k) / 30, num=len(data_30k)),
                    data_30k,
                    label=f'Channel {int(r*N_per_array+i)}')
    ax.set_title(f'Voltages')
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Channels')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig5b.pdf'))

# %%
with open(os.path.join(DATA_DIR, 'simulator', cursor_sim_latency_file),
          'rb') as f:
    latency_df = pd.read_csv(f)

# truncate at 60,000 samples and convert to millisecond units
latency_df = latency_df.iloc[:60_000, :] * 1e3

# %%
# Make plots
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7.5, 2.5))

# Cursor Simulator
axc = axes[1, :]
ts_labels = {
    'ts_fr': 'Firing Rate Model',
    'ts_thres': 'Spike Generation',
    'ts_cb_gen': 'Packet Generation'
}
ts_fields = ['ts_fr', 'ts_thres', 'ts_cb_gen']
labels = [ts_labels[field] for field in ts_fields]
# plot per-node latency as a histogram
step = 10e-3
for i, (label, field) in enumerate(zip(labels, ts_fields)):
    latency = latency_df[field].values
    bins = np.arange(latency.max() + step, step=step)
    axc[0].hist(latency,
                color=f'C{i + 1}',
                bins=bins,
                histtype='step',
                label=label)
axc[0].set_xlabel('Node Latency (ms)')
axc[0].set_ylabel('Samples')
axc[0].set_yscale('log')
axc[0].legend(fontsize=8, ncol=1, frameon=False, loc='upper right')

# plot cumulative latency as a horizontal violin plot
sns.violinplot(data=latency_df.cumsum(axis=1),
               scale='width',
               linewidth=0.2,
               orient='h',
               palette=sns_palette[1:],
               ax=axc[1])
axc[1].set_xlabel('Cumulative Latency (ms)')
axc[1].set_yticks(ticks=np.arange(latency_df.shape[1]), labels=labels)

# Speech Sim
axs = axes[0, :]
# load data
latency_dict = scipy.io.loadmat(
    os.path.join(DATA_DIR, 'speech_sim', 'mic_sim_latencies.mat'))

# Make plots
ts_labels = {
    'ts_fr': 'Firing Rate Model',
    'ts_thres': 'Spike Generation',
    'ts_cb_gen': 'Packet Generation'
}
ts_fields = ['ts_fr', 'ts_thres', 'ts_cb_gen']
labels = [ts_labels[field] for field in ts_fields]
# plot per-node latency as a histogram
step = 10e-3
latency = latency_dict['mfcc_latency'][0]
bins = np.arange(latency.max() + step, step=step)
axs[0].hist(latency, bins=bins, histtype='step', label='MFCC')

latency = latency_dict['frgen_latency'][0]
bins = np.arange(latency.max() + step, step=step)
axs[0].hist(latency, bins=bins, histtype='step', label='Firing Rate Model')

latency = latency_dict['fr30k_latency'][0]
bins = np.arange(latency.max() + step, step=step)
axs[0].hist(latency, bins=bins, histtype='step', label='Spike Generation')

latency = latency_dict['cbgen_latency'][0]
bins = np.arange(latency.max() + step, step=step)
axs[0].hist(latency, bins=bins, histtype='step',
            label='Packet Generation')  #\n& Buffering')

axs[0].set_xlabel('Node Latency (ms)')
axs[0].set_ylabel('Samples')
axs[0].set_yscale('log')
axs[0].legend(fontsize=8, ncol=1, frameon=False, loc='upper right')

# plot cumulative latency as a horizontal violin plot
sns.violinplot(data=[
    latency_dict['mfcc_latency'][0],
    latency_dict['mfcc_latency'][0] + latency_dict['frgen_latency'][0],
    latency_dict['mfcc_latency'][0] + latency_dict['frgen_latency'][0] +
    latency_dict['fr30k_latency'][0],
    latency_dict['mfcc_latency'][0] + latency_dict['frgen_latency'][0] +
    latency_dict['fr30k_latency'][0] + latency_dict['cbgen_latency'][0]
],
               scale='width',
               linewidth=0.2,
               orient='h',
               palette=sns_palette,
               ax=axs[1])
axs[1].set_xlabel('Cumulative Latency (ms)')
axs[1].set_yticks(ticks=[0, 1, 2, 3])
axs[1].set_yticklabels(
    ['MFCC', 'Firing Rate Model', 'Spike Generation', 'Packet Generation'])

# make the x-axs match
nplots = len(axes.ravel())
xlims = [axes.ravel()[ip].get_xlim()[1] for ip in range(nplots)]
for ip in range(nplots):
    axes.ravel()[ip].set_xlim(0, max(xlims))

# format figure and save
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig5d-g.pdf'))
# %%
