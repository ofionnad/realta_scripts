#!/usr/bin/env python3
import sys
import time 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import datetime as dt
import astropy.units as u
from sunpy.time import TimeRange
import xarray as xr

from scipy.signal import stft, istft
import scipy.stats as st


class LofarRaw:
    """
    Class attributed to a raw dataset from iLOFAR for plotting and analysis.

    :param fname: The filename
    :param sbs: A numpy array of the subbands in the data (e.g. np.arange(76, 320))
    :param obs_mode: The number of the ILOFAR observation mode (default=3)
    :param trange: Time range of observation, does noting right now (usually sunpy timerange object)
    :param frange: List of the minimum and maximum freqeuncies (e.g. [15, 60])
    :param bit_mode: Bit mode of the ILOFAR observation (default=8)
    :param d_type: dtype used to read the data from file using np.memmap (default=np.float32)
    :param dt: The time resolution of your data (Nyquist dt = 5.12e-6 * D, where D is decimation rate) 

    Hopefully will incorporate some spectral kurtosis functions that are efficient.
    """

    def __init__(self, fname, sbs, obs_mode, trange=None, frange=None, 
                bit_mode=8, d_type=np.float32, dt=5.12e-6):
        """
        Initialise the raw data object
        """
        self.fname = fname
        self.sbs = sbs
        self.obs_mode = obs_mode
        self.trange = trange
        self.frange = frange
        self.bit_mode = bit_mode
        self.d_type = d_type
        self.dt = dt
        self.freqs = self.sb_to_f(self.sbs, self.obs_mode)
        self.get_data()
        #self.data = self.data 

    def __repr__(self):
        return "iLofar mode {} observation".format(self.obs_mode) \
            + "Frequency range: " + str(self.freqs[0])


    @staticmethod
    def sb_to_f(sbs, obs_mode):
        """
        Convert subband number to a frequency in MHz
        """
        nyq_dict = {3:1, 5:2, 7:3}
        nyq_zone = nyq_dict[obs_mode]
        clock_dict = {3:200, 4:160, 5:200, 6:160, 7:200} #MHz
        clock = clock_dict[obs_mode]
        nu = (nyq_zone-1. + sbs/512.) * (clock/2.)
        return nu * u.MHz

    @staticmethod
    def plot_spectrum_plt(data, flimits, imshowmin=5, imshowmax=95):
        """
        :param flimits: List of 2 numbers, minimum and maximum subbands (e.g. [20, 150])
        :param imshowmin: Minimium percentile level
        :param imshowmax: Maximum percentile level
        """
        print("Started plotting...")
        plt.figure(figsize=(10,6))
        plt.imshow(data.T, aspect='auto', origin='lower',
            vmin=np.nanpercentile(data.T, imshowmin), 
            vmax=np.nanpercentile(data.T, imshowmax),
            extent=[0, data.T.shape[1], flimits[0], flimits[1]])
 #                   min(self.freqs[flimits[0]:flimits[1]]).value, 
 #                   max(self.freqs[flimits[0]:flimits[1]]).value])

    def plot_stft_map(self, i):
        """
        i - particular index corresponding to individual subband stft
        stft_i - value returned from scipy.signal.stft()
                should be a 3d tuple if passing more than 1 subband to stft()
        """
        f, ax = plt.subplots(2)
        ax[0].pcolormesh(self.skt, self.skf, np.abs(self.sk[:,i,:]), shading='auto', 
                        vmin=np.percentile(np.abs(self.sk[:,i,:]), 5), 
                        vmax=np.percentile(np.abs(self.sk[:,i,:]), 95))
        ax[1].pcolormesh(self.skt, self.skf, np.abs(self.sk_masked[:,i,:]), shading='auto', 
                        vmin=np.percentile(np.abs(self.sk_masked[:,i,:]), 5), 
                        vmax=np.percentile(np.abs(self.sk_masked[:,i,:]), 95))

    def spectral_kurtosis(stft_arr):
        """
        Expects an stft array from scipy.signal.stft() with dimensions (freq, subband, time)

        SK = <STFT>**4 / <STFT**2>**2 - 2.
        """
        frac_top = np.mean((np.abs(stft_arr)**4), axis=2)
        frac_bottom = np.mean((np.abs(stft_arr)**2), axis=2)**2
        return (frac_top / frac_bottom) - 2.

    def get_data(self):

        #read in data using numpy memory map
        s = time.time()
        data = np.memmap(self.fname, self.d_type, mode="c").reshape(-1, self.sbs.shape[0])
        data = np.flip(data, axis=1)
        data_read_time = time.time()
        print("Time to read data: {:.6f}s\n".format(data_read_time-s))

        #default time axis data
        time_res = str(self.dt / 1.e-6) + 'U' # U = microseconds
        self.times = pd.timedelta_range(start="0 millisecond", periods=data.shape[0], freq=time_res)
        #self.freqs = sb_to_f(sbs, obs_mode) #get list of frequencies

        self.data = data 

    def get_dark(self, fname, d_type, sbs_reshape, norm_from_file=False, write_to_file=False):
        s = time.time()
        dark = np.memmap(fname, d_type, mode="c").reshape(-1, sbs_reshape.shape[0])
        dark = np.flip(dark, axis=1)
        dark_read_time = time.time()
        print("Time to read dark data: {:.6f}s\n".format(dark_read_time-s))
        if norm_from_file:
            norms = np.fromfile('frequency_response_norm_dark.dat')
            print("Normalisation parameters read")
        else:
            norms = np.quantile(dark, 0.1, axis=0)
            print("Normalisation parameters calculated")
            norms.tofile('frequency_response_norm_dark.dat')
        dark = dark / norms
        print("Dark data normalisation done")
        self.dark = dark


    def data_normalise(self, dark_obs=False, use_xarray=True, read_norms=False, title=None):
        #You can use xarray or just numpy, depending on the datatype and size/shape
        if read_norms:
            s = time.time()
            norms = np.fromfile(title+'_frequency_response_normalisation.dat')
            self.data = self.data / norms
            print("Time to normalise data from file: {:.3f}s".format(time.time()-s))

        else:

            if use_xarray:
                dx = xr.DataArray(self.data, coords=[np.arange(len(self.times)),self.freqs], dims=["Time", "Frequency"])
                # Normalise data (removes frequency dependence)
                s = time.time()
                norms = dx.quantile(0.1, dim="Time")
                if write_response:
                    np.array(norms).tofile(title+'_frequency_response_normalisation.dat')
                self.data = dx/norms
                #data_mean = self.data.quantile(0.5, dim="Time")
                #self.sorted_freqs = data_mean.sortby(data_mean)
                print("Time to normalise data: {:.3f}s".format(time.time()-s))

            else:
                # Normalise data (removes frequency dependence)
                s = time.time()
                norms = np.quantile(np.abs(self.data), 0.1, axis=0) #calculate 10% values along time so we can normalise frequency response
                norms.tofile(title+'_frequency_response_normalisation.dat')
                self.data = self.data/norms
                print("Time to normalise data: {:.3f}s".format(time.time() - s))

        if dark_obs:
            s = time.time()
            self.data = self.data / self.dark
            print("Time to normalise with dark obs: {:.3f}s".format(time.time() - s))

    def apply_mask(self, data, rfi_bands):
        data_mask = np.zeros_like(data) #make an empty mask
        data_mask[:,rfi_bands] = 1 #mask the columns with rfi in them
        data_masked = np.ma.masked_array(data, data_mask) #apply mask to the data
        print("RFI dominated channels masked")
        return data_masked

    def sk_threshold(self, ci, sft_times):
        """Calculates the confidence interval for non-gaussian RFI in the STFT"""
        sd = np.sqrt(4. /sft_times) #variance is 4/length of time dimension
        return st.norm.interval(ci, scale=sd)[1] #returns coordinates that contain ci% of data in line with a gaussian centred at 0 with std=sd

    def do_stft(self, list_of_bands, ci=0.95, wind_div=50., skips=None):
        """
        This is where I tried to implement a SK algorithm just by using STFT.
        Beware that it could be slow. I try to circumvent this by selecting the subbands
        with the highest means and only putting those through the SK. Hoping that any RFI
        is very strong. We will see how this works out, might need to be tweaked in the future.
        
        :param ci: Confidence interval of non-gaussian noise to remove (default 0.95)
        
        """
        # Get the Short Time Fourier Transform of each subband (just 3 for now)
        if list_of_bands:
                dt = self.masked_data[:, list_of_bands]
                print('Performing STFT and clean on bands: {}\n'.format(list_of_bands))
        else:
            dt = self.masked_data
            print('Performing STFT on all data')

        print("stft input data shape", dt.shape)

        #get window size
        wind = int(self.data.shape[0] / wind_div)
        print("Window size: {}\n".format(wind))

        s = time.time()
        #from scipy.signal
        stft_f, stft_t, stft_data = stft(dt, window='hamming', axis=0, nperseg=wind) #nperseg=564 nfft=2048
        print("Time to do STFT for SK: {:.3f}s\n".format(time.time()-s))

        print("STFT data:", stft_data.shape, stft_data[0])

        #sk = spectral_kurtosis(sft[2])
        self.sk = stft_data
        self.skf = stft_f
        self.skt = stft_t
        limit = self.sk_threshold(ci, self.sk.shape[0])
        print('\n', limit, '\n')

        #mask the non-gaussian noise
        print("\nMasking non-gaussian noise...")
        stft_masked = np.ma.masked_outside(self.sk, v1=-limit, v2=limit, copy=False)
        self.sk_masked = stft_masked
        print("Done\n")

        #invert STFT back to data
        if list_of_bands:
            cleaned_data = []
            for i in range(self.sk.shape[1]):
                #note the settings here must be the same as the stft above
                _, cleaned_sbs = istft(stft_masked[:,i,:], window='hamming', nperseg=wind)
                cleaned_data.append(cleaned_sbs)

            #so now there is a list of subbands that have been cleaned
            #need to replace these back into original dataset
            
            #ensure the data is the same length in time dimension, pad the array
            cleaned_diff = self.data.shape[0] - np.array(cleaned_data).shape[1]

            #print(cleaned_diff)
            #print(cleaned_data[0].shape)

            for i,j in enumerate(list_of_bands):
                cleaned_padded = np.pad(cleaned_data[i], (int(cleaned_diff/2), 
                            int(cleaned_diff/2)+1), axis=1, mode='linear_ramp',
                            end_values=0)
                self.data[:j] = cleaned_padded[i] #replace the subbands with cleaned data.
                print("Band {} cleaned. \n".format(j))

        else:
            print("Inverse STFT...")
            cleaned_data_shape_time = (self.sk.shape[0]*self.sk.shape[2])-self.sk.shape[0] +1
            nchan = self.sk.shape[1]
            print(cleaned_data_shape_time, nchan)
            self.clean_data = np.empty([cleaned_data_shape_time, nchan])
            print(stft_masked.shape)
            for i in range(nchan):
                if skips is not None:
                    if i in skips:
                        self.clean_data[:,i].fill(np.nan)
                else:                       
                    _, cleaned_data = istft(stft_masked[:,i,:], window='hamming', nperseg=wind)
                    self.clean_data[:,i] = cleaned_data
                    print("Band # {} done.".format(i))
            #self.clean_data = np.array(self.clean_data)
            print("Cleaned data shape: {}".format(self.clean_data.shape))
            print("Done.\n")
        




if __name__=="__main__":
    """
    Run the class above
    """

    start = time.time()
    filename = sys.argv[1]

    #observation parameters
    frange = [15, 60]
    #fname = 'udpoutput/jupiter-stokesI_0_2020-10-13T17:47:00_19563125244140'
    sbs = np.arange(76, 320)
    beam1_sbs = np.arange(76, 198)
    obs_mode = 3
    trange = TimeRange("2020-10-13T17:45:00", 15.*u.min)
    """
    fname = 'udpoutput/jupiter-stokesI_0_2020-10-13T17:47:00_19563125244140'
    sbs = np.arange(76, 319)
    obs_mode = 3
    trange = TimeRange("2020-10-13T17:47:00", 10.*u.minute)
    """
    #just plotting parameters
    xlabel = "Time from 2020/10/13 17:47:00.0000"
    ylabel = "Frequency (MHz)"
    title = sys.argv[1].split('/')[1]

    #  instantiate data object and read in data
    raw_data = LofarRaw(fname=filename, sbs=sbs, obs_mode=obs_mode, frange=frange)
    #raw_data.get_dark("udpoutput/dark-stokesI_0_2020-10-13T17:47:00_19563125244140", np.float32, beam1_sbs, norm_from_file=True)

    flimits = np.array([35,100])  #65 sub bands
    freqlimits = raw_data.sb_to_f(flimits+sbs[0], obs_mode) #get the real frequencies of the plots
    print("Data range limited to: ", flimits, freqlimits)

    #remove RFI dominated subbands
    raw_data.data = raw_data.data[:,flimits[0]:flimits[1]] #cut out the higher frequencies

    #normalise frequency response
    raw_data.data_normalise(dark_obs=False, use_xarray=False, read_norms=False, title=title)

    
    raw_data.plot_spectrum_plt(data=raw_data.data, flimits=[freqlimits[0].value, freqlimits[1].value])
    plt.title(title + ' - raw data clipped')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(title+'.png')
        

    rfi_bands = [1, 3, 4, 11] #dominant rfi bands 0->20 + 31, 32, 38
    rfi_freqs = raw_data.sb_to_f(np.array(rfi_bands) + flimits[0] + sbs[0], obs_mode)
    print("\nMasked subbands: {}\nMasked frequencies: {}\n".format(np.array(rfi_bands)+flimits[0], rfi_freqs))

    #raw_data.data = raw_data.apply_mask(raw_data.data, rfi_bands)
    data_mask = np.zeros_like(raw_data.data) #make an empty mask
    data_mask[:,rfi_bands] = 1 #mask the columns with rfi in them
    raw_data.masked_data = np.ma.masked_array(raw_data.data, data_mask, copy=True) #apply mask to the data
    print("RFI dominated channels masked")

    
    raw_data.plot_spectrum_plt(data=raw_data.masked_data, flimits=[freqlimits[0].value, freqlimits[1].value])
    plt.title(title + ' - mask bad rfi channels')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(title+'masked.png')
    

    #do a stft for spectral kurtosis
    #limit = sk_threshold(ci=0.95, self.raw_data.data.shape[0])

    raw_data.do_stft(None, ci=0.5, wind_div=300., skips=rfi_bands) #do all the bands, with 100 windows, and 0.95 confidence interval
    raw_data.clean_data.ravel()
    print(raw_data.clean_data.shape)
    print(raw_data.clean_data[0].shape)

    
    raw_data.plot_spectrum_plt(data=np.abs(raw_data.clean_data), flimits=[freqlimits[0].value, freqlimits[1].value])
    plt.title(title + ' - cleaned with FT')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(title+'cleaned_sk.png')
    

    raw_data.clean_data.tofile('cleaned_data_'+title+'.dat')
    print("Total time: {:.3f}s".format(time.time()-start))

    #lets add a plot of one of the stft channels to see what it looks like
    raw_data.plot_stft_map(60)
    plt.title(title + ' -STFT- channel: 60')
    plt.ylabel("Frequencies")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(title+'stft_sk_image.png')
