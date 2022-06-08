#!/usr/bin/env python

import numpy as np
import sys

import datetime as dt
from sunpy.time import TimeRange
import astropy.units as u

#plotting modules
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib import gridspec,transforms

#custom module
sys.path.insert(1, '../../20201013_jupiter/')
from sk import LofarRaw

def get_data_lims(sbs, obs_mode, sbs_number, trange):
    """
    Return the limits of the data in usable form
    sbs_number = number of subbands used
    trange = the length of time. 

    """

    flimits = np.array([0,sbs_number])
    freqlimits = LofarRaw.sb_to_f(flimits+sbs[0], obs_mode)
    xlims = list(map(dt.datetime.fromisoformat, [trange.start.value, trange.end.value]))
    xlims = mdates.date2num(xlims)

    return freqlimits, xlims

def plot_data(data, xs, ys, xlims, flims, xlabel, ylabel, plotname, plot_title, minutes=True, gs=False):
    print("\nPlotting...\n")
    if gs:
        cmap = plt.cm.Greys
    else:
        cmap = plt.cm.viridis

    if xs is None and ys is None:
        print("Not using summed axes!")
        f, ax = plt.subplots(figsize=(12,6))
        f.set_facecolor('white')
        im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
            vmin=np.nanpercentile(data.T, 5), 
            vmax=np.nanpercentile(data.T, 90),
            extent=[xlims[0], xlims[1], 
                    flims[0].value, flims[1].value])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)

        ax.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M:%S.%f')
        ax.xaxis.set_major_formatter(date_format)
        if minutes:
            ax.xaxis.set_minor_locator(mdates.MinuteLocator())
        f.autofmt_xdate()

        ax.set_title(plot_title)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)

        plt.tight_layout()
        plt.savefig(plotname, dpi=900)

    else:

        f = plt.figure(figsize=(12,6))
        f.set_facecolor('white')

        spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[6,1], height_ratios=[1,4])
        ax0 = f.add_subplot(spec[0])
        ax2 = f.add_subplot(spec[2])
        ax3 = f.add_subplot(spec[3])

        ax0.plot(xs, lw=0.5)
        ax0.set(xticklabels=[], xlim=[0,None])
        ax0.tick_params(bottom=False)

        rot = transforms.Affine2D().rotate_deg(270)
        ax3.plot(ys[::-1], lw=0.5, transform=rot+plt.gca().transData)
        ax3.set(yticklabels=[])
        ax3.tick_params(left=False)

        im = ax2.imshow(data.T, aspect='auto', origin='lower',
                vmin=np.nanpercentile(data.T, 5), 
                vmax=np.nanpercentile(data.T, 90),
                extent=[xlims[0], xlims[1], 
                        flims[0].value, flims[1].value])

        ax2.xaxis_date()
        date_format = mdates.DateFormatter('%H:%M:%S.%f')
        ax2.xaxis.set_major_formatter(date_format)
        if minutes:
            ax2.xaxis.set_minor_locator(mdates.MinuteLocator())
        f.autofmt_xdate()
        
        ax2.set_title("Uranus observation - Stokes I")
        ax2.set_ylabel(ylabel, fontsize=14)
        ax2.set_xlabel(xlabel, fontsize=14)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.savefig(plotname, dpi=900)

def remove_rfi_wtih_sd(data, stdy=2, stdx=2):
    """
    Function to mask out suspected rfi automatically

    data - data to be masked
    stdy - mask threshold for channels in number of standard deviations
    stdx - mask threshold for time in number of standard deviations

    Returns the masked data and the result of the sum along each axis
    """
    print('\nRemoving RFI...\n')

    #sum along the time axis
    ys = data.T.sum(axis=1)
    sigma_y = stdy*np.std(ys) #2sigma
    ymean = np.mean(ys)
    #print(ymean+sigma_y)

    for i,j in enumerate(ys[::-1]):
        rfi_removed_list = []
        if j > (ymean+sigma_y):
            rfi_removed_list.append(i)
            ys[i] = np.NaN 
            data[:, i] = np.NaN
    if rfi_removed_list is not None:
        print("RFI trigger channel : {}".format(rfi_removed_list))

    #ys = data.T.sum(axis=1)
    xs = np.nansum(data.T, axis=0)
    sigma_x = stdx*np.nanstd(xs) #2sigma
    xmean = np.nanmean(xs)
    #print(xmean + sigma_x)
    for i,j in enumerate(xs):
        rfi_removed_list = []
        if j > (xmean+sigma_x):
            xs[i] = np.NaN
            data[i, :] = np.NaN
    if rfi_removed_list is not None:
        print("RFI trigger time sample: {}".format(rfi_removed_list))

    return xs, ys, data 

def resample_dataset(data, f=12207):
    """
    Function takes the dataset, find the remainder after dividing by the resample factor (f) and pads the array. 
    Then takes the mean of the chunks of data of those resampled size. 

    f - resampling factor 
    """
    pad = np.ceil(float(data.shape[0])/f)*f - data.shape[0]
    pad_arr = np.zeros((int(pad), data.shape[1]))*np.NaN
    data_padded = np.append(data, pad_arr).reshape(-1, data.shape[1])
    r_data = np.nanmean(data_padded.T.reshape(data.shape[1], -1, f), axis=2).T

    print('\tData downsampled x{}'.format(int(f)))
    print('\tData shape: {}'.format(r_data.shape))

    return r_data

def data_chunker(df, n_split):
    """
    Function to split up data
    df - dataset to be split
    n_split - number of splits
    """
    chunked = np.array_split(df, n_split, axis=0)

    return chunked

def frequency_response_normalisation(data):
    """
    Function to normalise the frequency response trend across channels
    """
    ys = np.nansum(data, axis=0)
    print(ys.shape)
    ys_x = np.linspace(0, ys.shape[0], ys.shape[0])
    smooth_fit = np.poly1d(np.polyfit(ys_x, ys, 2))
    smooth_fresponse = smooth_fit(ys_x)
    norm_data = data / smooth_fresponse

    return norm_data

def bg_normalisation(data, bg_data):
    """
    Returns the data normalised by the background
    """
    norm_data = np.true_divide(data, bg_data, out=np.zeros_like(data), where=bg_data[i]!=0)
    return norm_data

if __name__=="__main__":

    #observation specs
    # filename = '../udpoutput/uranus-stokesVectors_0_2020-12-15T20:04:00_19629670898060' #0 is I and 1 is V I assume
    #off_fname = '../udpoutput/offsource-stokesVectors_0_2020-12-15T20:04:00_19629670898060'
    #cal_fname = '../udpoutput/cygA-stokesVectors_0_2020-12-15T20:00:00_19629667968374' #calibrator has different obs length!

    filename = '/mnt/ucc4_data1/data/Uranus/2021_11_30/Uranus_IE613_2021-11-30T23:00:00.000.fil'

    plot_names = '../data/20211130_uranus/all_data_plots/Uranus_StokesI_'
    plot_title = "Uranus observation - Stokes I - IE613 - 2021-11-30"

    frange = [15,50]
    sbs = np.arange(40,451)
    obs_mode = 3
    time_len_mins = 60.
    trange = TimeRange(filename.split('_')[-1].strip('.fil'), time_len_mins*u.min)
    xlabel = "Time from {} {}".format(filename.split('_')[-1].strip('.fil').split('T')[0], filename.split('_')[-1].strip('.fil').split('T')[1])
    xlabel = "Time on {} (UTC)".format(filename.split('_')[-1].strip('.fil').split('T')[0])
    ylabel = "Frequency (MHz)"
    title = filename.split('/')[-1]
    no_sbs = 411 #number of usable subbands!

    #how much to split up data into
    nsplit = 10
    r_factor = 64

    #on-beam
    rawdata = LofarRaw(fname=filename, sbs=sbs, obs_mode=obs_mode, frange=frange)
    rawdata.data = rawdata.data[:,:no_sbs] #need to do this because of the way subbands were set up for uranus observations! (only use 78 subbands!)
    #off-beam
    # rawoffdata = LofarRaw(fname=off_fname, sbs=sbs, obs_mode=obs_mode, frange=frange)
    # rawoffdata.data = rawoffdata.data[:,:no_sbs]

    ylims, xlims = get_data_lims(sbs, obs_mode, no_sbs, trange)

    df_chunk = data_chunker(rawdata.data, nsplit)
    dts = trange.split(nsplit)
    # off_chunk = data_chunker(rawoffdata.data, nsplit)
    
    for i,j in enumerate(df_chunk):
        print(plot_names+str(i))
        temp = resample_dataset(j, f=r_factor)
        # offtemp = resample_dataset(off_chunk[i], f=r_factor)
        #temp = frequency_response_normalisation(temp)
        # temp = bg_normalisation(temp, offtemp)
        xs, ys, temp = remove_rfi_wtih_sd(temp)
        #print(ylims)
        plot_data(temp, xs, ys, trange, xlims, ylims, xlabel, ylabel, plot_names+str(i), plot_title)