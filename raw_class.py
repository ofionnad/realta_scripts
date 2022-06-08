import copy
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time, TimeDelta
from astropy.visualization import AsymmetricPercentileInterval
from matplotlib import dates
from sunpy.time import TimeRange

def sb_to_freq(sb, obs_mode):
        """
        Converts LOFAR single station subbands to frequency
        Returns frequency as astropy.units.Quantity (MHz)
        Inputs: subband number, observation mode (3, 5 or 7)
        """
        nyq_zone_dict = {3:1, 5:2, 7:3}
        nyq_zone = nyq_zone_dict[obs_mode]
        clock_dict = {3:200, 4:160, 5:200, 6:160, 7:200} #MHz
        clock = clock_dict[obs_mode]
        freq = (nyq_zone-1+sb/512)*(clock/2)
        return freq*u.MHz #MHz


class LofarRaw:
	"""
	class that reads and plots the data from a raw ilofar file. 
	This is usually the output from udpPM

	"""

	def __init__(self, fname, sbs, obs_mode, trange=None, frange=None, bit_mode=8, dtype=np.float32, dt=5.12e-6):
		"""
		description of variables passed to this class

		fname : raw filename
		sbs : a numpy list of subbands (required for correct frequency ranges)
		obs_mode : the observational mode ilofar was used in
		trange : The time range that the observations were made (sunpy TimeRange object)
		frange : minimum and maximum frequecnies of observation (in astropy MHz units)
		bit_mode : observations bit mode, default 8. 
		dtype : data type for numpy memory allocation
		dt : time resolution (careful with time decimation)
		"""

		self.fname = fname
		self.sbs = sbs
		self.obs_mode = obs_mode
		self.trange = trange
		self.frange = frange
		self.bit_mode = bit_mode
		self.dtype = dtype
		self.dt = TimeDelta(dt, format="sec")
		self.obs_start = Time(fname.split("_")[2])
		self.freqs = sb_to_freq(self.sbs, self.obs_mode)
		self.__get_data()
		self.obs_end = self.obs_start + (self.dt*self.data.shape[1])


	def __get_data(self):
		"""
		Extract the data from the raw binary file.
		"""

		### if the time range is specified
		if self.trange is not None:
			dtype_ratio = {np.float32:4, np.int8:1}
			offset_time = self.trange.start - self.obs_start #if trange and obstart are the same this is just zero
			offset = np.floor(offset_time/self.dt) * len(self.sbs) * dtype_ratio[self.dtype]
			offset = int(offset)

			#count is the number of data points in the time range
			# total time / delta time * # of subbands
			count = np.floor(self.trange.seconds.value/self.dt.sec) * len(self.sbs)
			count = int(count)

			#now use numpy memory mapping to make a numpy array that points to the right memory
			#this function allows data to be read in in chunks instead of all at once
			data = np.memmap(self.fname, self.dtype, shape=count, offset=offset, mode="c").reshape(-1, self.sbs.shape[0])

		#if timerange is not specified:
		else:
			data = np.memmap(self.fname, self.dtype, mode="c").reshape(-1, self.sbs.shape[0])
			self.freqs = sb_to_freq(self.sbs, self.obs_mode)


		#now to deal with the frequency range:
		#freqs = sb_to_freq(self.sbs, self.obs_mode) ##defined above
		if self.frange is not None:
			f_start_end = []
			for freq in self.frange:
				diff_arr = np.abs(freq.to(u.MHz) - self.freqs)
				min_val = np.min(diff_arr)
				f_start_end.append(np.where(diff_arr == min_val)[0][0])
			f_start, f_end = f_start_end

		#if frequency range is not specified just do 0 to max
		else:
			f_start = 0
			f_end = None

		self.freqs = self.freqs[f_start:f_end]

		data = data[:,f_start:f_end].T #put the frequency first
		self.data = np.flip(data, axis=0) #flip the frequency axis

