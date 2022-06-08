import astropy.units as u
from astropy.time import Time
import sigpyproc as spp
import pandas as pd
import time
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec,transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
from sunpy.time import TimeRange

class LofarFilterBank():

    def __init__(self, fname, sbs, obs_mode, tstart, trange=None, frange=None, 
                bit_mode=8, stokes=False, d_type=np.float32, t_res=1, tlimits=None):

        """
        Initialise the filterbank data object
        """
        self.fname = fname
        self.sbs = sbs
        self.obs_mode = obs_mode
        self.trange = trange
        self.frange = frange
        self.stokes = stokes
        self.bit_mode = bit_mode
        self.d_type = d_type
        self.t_res = t_res
        self.freqs = self.sb_to_f(self.sbs, self.obs_mode)
        if tlimits:
            self.make_subset(tlimits)
        else:
            self.get_data(tstart)
        self.dt = self.datareader.header.tsamp

    def __repr__(self):
        return "LOFAR mode {} observation".format(self.obs_mode) \
            + "\nFrequency range: " + str(self.freqs[0]) + "\nTime: " + str(self.trange)


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

    def get_data(self, tstart):
        """
        tstart is a string of the starting time
        """

        #read in data using sigprocpy filterbank
        s = time.time()
        self.datareader = spp.FilReader(self.fname)
        dataBlock_all = self.datareader.readBlock(0, self.datareader.header.nsamples)
        data = spp.Filterbank.FilterbankBlock(dataBlock_all, dataBlock_all.header)

        data_read_time = time.time()
        print("Time to read data: {:.6f}s\n".format(data_read_time-s))

        #default time axis data
        time_len = data.shape[1]

        self.dt = self.datareader.header.tsamp
        self.trange = TimeRange(tstart, time_len*self.dt*u.second)
        self.freqs = self.sb_to_f(self.sbs, self.obs_mode) #get list of frequencies

        self.data = data
        self.datablock = dataBlock_all

    def do_normalise(self):
        self.data = self.data.normalise()
        print("Data normalised")

    def do_downsample(self, t_res=None):
        if t_res:
            dsample = t_res/self.dt
        else:
            dsample = self.t_res/self.dt
        print("Downsampling by {}".format(int(dsample)))
        self.data = self.data.downsample(tfactor=int(dsample))
        print("Data downsampled to {}".format(self.dt*dsample))

    def subtract_background(self):
        bg = np.array([10**(np.mean(np.log10(f))) for f in self.data])
        self.data /= bg
        self.data = 20*np.log10(self.data)

    def spectral_kurtosis(self):
        """
        Expects an stft array from scipy.signal.stft() with dimensions (freq, subband, time)

        SK = <STFT>**4 / <STFT**2>**2 - 2.
        """
        frac_top = np.mean((np.abs(self.stft)**4), axis=2)
        frac_bottom = np.mean((np.abs(self.stft)**2), axis=2)**2
        self.sk = (frac_top / frac_bottom) - 2.
        return self.sk

    def do_stft(self, fs, nperseg):
        self.xt, self.yf, self.stft = scipy.signal.stft(self.data, fs=fs, nperseg=nperseg)

    def plot_stft(self, fpath):
        plt.imshow(self.stft, origin='lower', extent=[0,self.xt[-1],0,self.yf[-1]])
        plt.tight_layout()
        plt.savefig(fpath)

    def remove_sk(self, std_dev, fs, nperseg):
        self.sk_masked = np.ma.masked_array(self.sk)
        for i, row in enumerate(self.sk):
            st = std_dev*row.std()
            self.sk_masked[i, row > st] = np.ma.masked

    def make_subset(self, tlimits):
        self.datareader = spp.FilReader(self.fname)
        t1 = Time(datetime.datetime.strptime(tlimits[0], "%Y%m%d%H%M%S"), format='datetime')
        t2 = Time(datetime.datetime.strptime(tlimits[1], "%Y%m%d%H%M%S"), format='datetime')
        t1.format = "mjd"
        t2.format = "mjd"
        time_file = Time(self.datareader.header.tstart, format = "mjd")
        tdelta = (t1 - time_file).sec
        tdelta2 = (t2 - t1).sec
        if not self.stokes:
            factor=4
        else:
            factor=1
        readTimestamp = int(tdelta/self.datareader.header.tsamp)*factor
        samplesPerBlock = int(tdelta2/self.datareader.header.tsamp)*factor
        self.data = self.datareader.readBlock(readTimestamp, samplesPerBlock)
        self.trange = TimeRange(t1, t2)
        # self.data = self.data[flimits[0]:flimits[1],tlimits[0]:tlimits[1]]

    def plot_data(self, xlabel=None, ylabel=None, plotname='test.png', plot_title='title', minutes=True, gs=False, stft=False, sk=False, chan=50):
        print("\nPlotting...\n")
        if gs:
            cmap = plt.cm.Greys
        else:
            cmap = plt.cm.viridis

        if stft:
            data = self.stft.real[chan,:,:].T
            ys=None
            xs=None
            flims = [self.yf[0], self.yf[-1]]
            xlims = [self.xt[0], self.xt[-1]]
        elif sk:
            data = self.sk.T
            ys=None
            xs=None
            flims = [self.yf[0], self.yf[-1]]
            xlims = [self.xt[0], self.xt[-1]]
        else:
            data = self.data.T
            data = np.flip(data, axis=1)
            ys = data.sum(axis=0)
            xs = data.sum(axis=1)
            flims, xlims = self.get_data_lims()

        if xs is None and ys is None:
            print("Not using summed axes!")
            f, ax = plt.subplots(figsize=(12,6))
            f.set_facecolor('white')
            if sk or stft:
                im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                    vmin=np.nanpercentile(data.T, 5), 
                    vmax=np.nanpercentile(data.T, 90),
                    extent=[xlims[0], xlims[1], 
                        flims[0], flims[1]])
            else:
                im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
                    vmin=np.nanpercentile(data.T, 5), 
                    vmax=np.nanpercentile(data.T, 90),
                    extent=[xlims[0], xlims[1], 
                        flims[0].value, flims[1].value])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax)

            if not stft and not sk:
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
            plt.savefig(plotname, dpi=400)

        else:

            f = plt.figure(figsize=(12,6))
            f.set_facecolor('white')

            spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[6,1], height_ratios=[1,4])
            ax0 = f.add_subplot(spec[0])
            ax2 = f.add_subplot(spec[2])
            ax3 = f.add_subplot(spec[3])

            ax0.plot(np.arange(xs.size), xs, lw=0.5)
            ax0.set(xticklabels=[], xlim=[0,xs.size-1])
            ax0.tick_params(bottom=False)

            # rot = transforms.Affine2D().rotate_deg(90)
            # ax3.plot(ys, lw=0.5, transform=rot+plt.gca().transData)
            ax3.set(yticklabels=[])
            ax3.tick_params(left=False)
            ax3.plot(ys, np.arange(ys.size))
            #xtmp = ax3.get_xlim()
            #ax3.set_xlim([np.nanmin(ys)*0.95, np.nanmax(ys)*1.05])
            #ax3.set_ylim(xtmp)
            # ax3.invert_yaxis()
            # ax3.invert_xaxis()

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
            plt.savefig(plotname, dpi=400)

    def get_data_lims(self):
        """
        Return the limits of the data in usable form
        sbs_number = number of subbands used
        trange = the length of time. 

        """

        flimits = np.array([0,self.sbs[-1]-self.sbs[0]+1])
        freqlimits = self.sb_to_f(flimits+self.sbs[0], self.obs_mode)
        xlims = list(map(datetime.datetime.fromisoformat, [self.trange.start.iso, self.trange.end.iso]))
        xlims = mdates.date2num(xlims)

        return freqlimits, xlims
