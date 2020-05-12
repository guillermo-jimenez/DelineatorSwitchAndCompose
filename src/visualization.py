import matplotlib.pyplot as plt
import numpy as np


def plot_signal(config, data, key, on, off=None, ax=None, xlabel=True, ylabel=True, twinx_label=True, labelsize=14, fontsize=24):
    if ax == None:
        ax = plt.gca()
    if off == None:
        off = on + config.window
    if config.sampling_freq == None:
        f = 1
    else:
        f = config.sampling_freq

    ax.plot(np.linspace(on/f,off/f,num=off-on),data.dataset[key][on:off],'b')
    ax.axvspan(on/f,off/f,facecolor='y', alpha=0.1)
    ax.set_xlim([on/f,off/f])
    ax.tick_params(axis='both', which='major', labelsize=labelsize, labelrotation=45.)

    if not(xlabel):
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (s)', fontsize=fontsize, fontname='Times New Roman')

    if not(ylabel):
        ax.set_yticks([])
    else:
        ax.set_ylabel('Voltage (mV)', fontsize=fontsize, fontname='Times New Roman')

    return ax


def plot_markers(config, data, key, on, off=None, ax=None, xlabel=True, ylabel=True, twinx_label=True, labelsize=14, fontsize=24, markersize=10):
    if ax == None:
        ax = plt.gca()
    if off == None:
        off = on + config.window
    if config.sampling_freq == None:
        f = 1
    else:
        f = config.sampling_freq

    ax.plot(np.linspace(on/f,off/f,num=off-on),data.dataset[key][on:off],'b')
    ax.plot(data.P.onset[key][(data.P.onset[key] >= on) & (data.P.onset[key] <= off)]/f, data.dataset[key][data.P.onset[key][(data.P.onset[key] >= on) & (data.P.onset[key] <= off)]],'r*',markersize=markersize)
    ax.plot(data.P.peak[key][(data.P.peak[key] >= on) & (data.P.peak[key] <= off)]/f, data.dataset[key][data.P.peak[key][(data.P.peak[key] >= on) & (data.P.peak[key] <= off)]],'r*',markersize=markersize)
    ax.plot(data.P.offset[key][(data.P.offset[key] >= on) & (data.P.offset[key] <= off)]/f, data.dataset[key][data.P.offset[key][(data.P.offset[key] >= on) & (data.P.offset[key] <= off)]],'r*',markersize=markersize)

    ax.plot(data.QRS.onset[key][(data.QRS.onset[key] >= on) & (data.QRS.onset[key] <= off)]/f, data.dataset[key][data.QRS.onset[key][(data.QRS.onset[key] >= on) & (data.QRS.onset[key] <= off)]],'g^',markersize=markersize)
    ax.plot(data.QRS.peak[key][(data.QRS.peak[key] >= on) & (data.QRS.peak[key] <= off)]/f, data.dataset[key][data.QRS.peak[key][(data.QRS.peak[key] >= on) & (data.QRS.peak[key] <= off)]],'g^',markersize=markersize)
    ax.plot(data.QRS.offset[key][(data.QRS.offset[key] >= on) & (data.QRS.offset[key] <= off)]/f, data.dataset[key][data.QRS.offset[key][(data.QRS.offset[key] >= on) & (data.QRS.offset[key] <= off)]],'g^',markersize=markersize)

    ax.plot(data.T.onset[key][(data.T.onset[key] >= on) & (data.T.onset[key] <= off)]/f, data.dataset[key][data.T.onset[key][(data.T.onset[key] >= on) & (data.T.onset[key] <= off)]],'mo',markersize=markersize)
    ax.plot(data.T.peak[key][(data.T.peak[key] >= on) & (data.T.peak[key] <= off)]/f, data.dataset[key][data.T.peak[key][(data.T.peak[key] >= on) & (data.T.peak[key] <= off)]],'mo',markersize=markersize)
    ax.plot(data.T.offset[key][(data.T.offset[key] >= on) & (data.T.offset[key] <= off)]/f, data.dataset[key][data.T.offset[key][(data.T.offset[key] >= on) & (data.T.offset[key] <= off)]],'mo',markersize=markersize)

    ax.axvspan(on/f,off/f,facecolor='y', alpha=0.0)
    ax.set_xlim([on/f,off/f])
    ax.tick_params(axis='both', which='major', labelsize=labelsize, labelrotation=45.)
    
    if not(xlabel):
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (s)', fontsize=fontsize, fontname='Times New Roman')

    if not(ylabel):
        ax.set_yticks([])
    else:
        ax.set_ylabel('Voltage (mV)', fontsize=fontsize, fontname='Times New Roman')

    return ax


def plot_mask(config, data, key, on, off=None, ax=None, xlabel=True, ylabel=True, twinx_label=True, labelsize=14, fontsize=24):
    if ax == None:
        ax = plt.gca()
    if off == None:
        off = on + config.window
    if config.sampling_freq == None:
        f = 1
    else:
        f = config.sampling_freq

    ax.plot(np.linspace(on/f,off/f,num=off-on),data.dataset[key][on:off],'b')

    # Plot mask
    ax_2 = ax.twinx()

    # Plot
    ax_2.plot(np.linspace(on/f,off/f,num=off-on),data.P.wave[key].values[on:off],'r-')
    ax_2.plot(np.linspace(on/f,off/f,num=off-on),data.QRS.wave[key].values[on:off],'g-')
    ax_2.plot(np.linspace(on/f,off/f,num=off-on),data.T.wave[key].values[on:off],'m-')

    # Fill
    ax_2.fill_between(np.linspace(on/f,off/f,num=off-on),data.P.wave[key].values[on:off],0,where=data.P.wave[key].values[on:off]>=np.zeros(data.P.wave[key].values[on:off].shape), color='red', alpha=0.1)
    ax_2.fill_between(np.linspace(on/f,off/f,num=off-on),data.QRS.wave[key].values[on:off],0,where=data.QRS.wave[key].values[on:off]>=np.zeros(data.QRS.wave[key].values[on:off].shape), color='green', alpha=0.1)
    ax_2.fill_between(np.linspace(on/f,off/f,num=off-on),data.T.wave[key].values[on:off],0,where=data.T.wave[key].values[on:off]>=np.zeros(data.T.wave[key].values[on:off].shape), color='magenta', alpha=0.1)

    ax.axvspan(on/f,off/f,facecolor='y', alpha=0.0)
    ax.set_xlim([on/f,off/f])
    ax.tick_params(axis='both', which='major', labelsize=labelsize, labelrotation=45.)
    
    if not(xlabel):
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (s)', fontsize=fontsize, fontname='Times New Roman')

    if not(ylabel):
        ax.set_yticks([])
    else:
        ax.set_ylabel('Voltage (mV)', fontsize=fontsize, fontname='Times New Roman')
    
    if (twinx_label):
        ax_2.set_yticks([0., 1.]) # Binary mask
        ax_2.set_ylabel('Arbitrary units', fontsize=fontsize, fontname='Times New Roman')
    else:
        ax_2.set_yticks([])

    return ax


def plot_all(config, data, key, on, off=None, ax=None, xlabel=True, ylabel=True, twinx_label=True, labelsize=14, fontsize=24):
    if ax == None:
        ax = plt.gca()
    if off == None:
        off = on + config.window
    f = config.sampling_freq

    ax.plot(np.linspace(on/f,off/f,num=off-on),data.dataset[key][on:off],'b')

    onsets = np.sort(data.P.onset[key][(data.P.onset[key] >= on) & (data.P.onset[key] <= off)])
    offsets = np.sort(data.P.offset[key][(data.P.onset[key] >= on) & (data.P.offset[key] <= off)])

    # If any wave found
    if len(offsets) > 0:
        if (len(onsets) != len(offsets)) & (onsets[0] > offsets[0]):
            onsets = np.asarray([on] + onsets.tolist(), dtype=int)
        elif (len(onsets) != len(offsets)) & (onsets[0] < offsets[0]):
            offsets = np.asarray(offsets.tolist() + [off], dtype=int)
        elif (onsets[0] > offsets[0]):
            onsets = np.asarray([on] + onsets.tolist(), dtype=int)
            offsets = np.asarray(offsets.tolist() + [off], dtype=int)

    for i in range(len(onsets)):
        ax.axvspan(onsets[i]/f, 
                    offsets[i]/f, 
                    facecolor='r', alpha=0.3)

    onsets = np.sort(data.QRS.onset[key][(data.QRS.onset[key] >= on) & (data.QRS.onset[key] <= off)])
    offsets = np.sort(data.QRS.offset[key][(data.QRS.onset[key] >= on) & (data.QRS.offset[key] <= off)])
    
    if len(offsets) > 0:
        if (len(onsets) != len(offsets)) & (onsets[0] > offsets[0]):
            onsets = np.asarray([on] + onsets.tolist(), dtype=int)
        elif (len(onsets) != len(offsets)) & (onsets[0] < offsets[0]):
            offsets = np.asarray(offsets.tolist() + [off], dtype=int)
        elif (onsets[0] > offsets[0]):
            onsets = np.asarray([on] + onsets.tolist(), dtype=int)
            offsets = np.asarray(offsets.tolist() + [off], dtype=int)
            
    for i in range(len(onsets)):
        ax.axvspan(onsets[i]/f, 
                    offsets[i]/f, 
                    facecolor='g', alpha=0.3)

    onsets = np.sort(data.T.onset[key][(data.T.onset[key] >= on) & (data.T.onset[key] <= off)])
    offsets = np.sort(data.T.offset[key][(data.T.onset[key] >= on) & (data.T.offset[key] <= off)])

    if len(offsets) > 0:
        if (len(onsets) != len(offsets)) & (onsets[0] > offsets[0]):
            onsets = np.asarray([on] + onsets.tolist(), dtype=int)
        elif (len(onsets) != len(offsets)) & (onsets[0] < offsets[0]):
            offsets = np.asarray(offsets.tolist() + [off], dtype=int)
        elif (onsets[0] > offsets[0]):
            onsets = np.asarray([on] + onsets.tolist(), dtype=int)
            offsets = np.asarray(offsets.tolist() + [off], dtype=int)
            
    for i in range(len(onsets)):
        ax.axvspan(onsets[i]/f, 
                    offsets[i]/f, 
                    facecolor='m', alpha=0.3)

    ax.axvspan(on/f,off/f,facecolor='y', alpha=0.05)
    ax.set_xlim([on/f,off/f])
    ax.tick_params(axis='both', which='major', labelsize=labelsize, labelrotation=45.)
    # ax.get_yaxis().set_label_coords(-0.09,0.5)

    if not(xlabel):
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (s)', fontsize=fontsize, fontname='Times New Roman')

    if not(ylabel):
        ax.set_yticks([])
    else:
        ax.set_ylabel('Voltage (mV)', fontsize=fontsize, fontname='Times New Roman')

    return ax
