#!/usr/bin/env python

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

class LOFAR_BST:
	"""
	Class to read and plot I-LOFAR beamformed statistics (BST) data
	Required arguments: bst_file = File name,
						sbs = subbands observed,
						obs_mode = observation mode
	Optional arguments: trange = time range (sunpy.time.TimeRange), 
						frange = frequency range (2 element list/array with start and end frequency as astropy.units.Quantity e.g [10,90]*u.MHz),
						bit_mode = bit mode (usually 4, 8 or 16),
						integration = integration time (seconds)

						Default values of bit_mode and integration are typical of I-LOFAR observations
	attributes: bst_file    = file name
				sbs         = subbands observed
				obs_mode    = observing mode
				trange      = specified timerange 
				frange      = specified frequency range
				bit_mode    = bit mode
				obs_start   = observation start time
				integration = integration time (seconds)
				data        = data (sbs x (total time/integration) array)
				times       = time array
				freqs       = frequency array

	"""
	def __init__(self,bst_file, sbs, obs_mode, trange=None, frange=None, bit_mode=8, integration=1):
		self.bst_file = bst_file
		self.sbs = sbs
		self.obs_mode = obs_mode
		self.trange = trange
		self.frange = frange
		self.bit_mode = bit_mode
		if len(file_split := bst_file.split("_bst")) > 1:
			obs_start = file_split[0][-15:]
			self.obs_start = Time.strptime(obs_start, "%Y%m%d_%H%M%S")
		else:
			#assume raw data of format "sun_stokes_datetime_packetnumber"
			self.obs_start = Time(file_split[0].split("_")[2])
		self.integration = TimeDelta(integration, format="sec")
		self.__get_data()
		self.data = self.data.T

	def __repr__(self):
		return "I-LOFAR mode {} observation".format(self.obs_mode) \
				+ "\n Time range:" + str(self.times[0]) + " to " + str(self.times[-1]) \
				+ "\n Frequency range:" + str(self.freqs[0]) + " to " + str(self.freqs[-1])

	def __get_data(self):
		"""
		Read in data from BST file and calculate time and frequency
		"""
		data = np.fromfile(self.bst_file)
		t_len = data.shape[0]/len(self.sbs)
		self.data = data.reshape(-1, len(self.sbs))
		tarr = np.arange(t_len) * self.integration
		self.times = self.obs_start + tarr
		if self.trange is not None:
			t_start_end = []

			for time in (self.trange.start, self.trange.end):
				diff_arr = np.abs(time-self.times)
				min_val = np.min(diff_arr)
				t_start_end.append(np.where(diff_arr == min_val.value)[0][0])
			t_start, t_end = t_start_end
		else:
			t_start = 0
			t_end = None
		self.times = self.times[t_start:t_end]
		self.freqs = sb_to_freq(self.sbs, self.obs_mode)
		if self.frange is not None:
			f_start_end = []
			for freq in self.frange:
				diff_arr = np.abs(freq.to(u.MHz)-self.freqs)
				min_val = np.min(diff_arr)
				f_start_end.append(np.where(diff_arr == min_val)[0][0])
			f_start, f_end = f_start_end
		else:
			f_start = 0
			f_end = None
		self.freqs = self.freqs[f_start:f_end]
		self.data = self.data[t_start:t_end, f_start:f_end]
	
	def bg(self, data, amount=0.05):
		#adapted from radiospectra.Spectrogram.auto_find_background()
		#doesn't do mean subtraction at the beginning
		tmp = data# - np.mean(data, 1)[:, np.newaxis]
		sdevs = np.std(tmp, 0)
		cand = sorted(range(data.shape[1]), key=lambda y: sdevs[y])
		realcand = cand[:max(1, int(amount*len(cand)))]
		bg = np.mean(data[:,realcand], 1)
		return bg[:, np.newaxis]
	
	def plot(self, ax=None, title=None, bg_subtract=False,
				   scale="linear", clip_interval: u.percent=None):
		"""
		Plot dynamic spectrum for whole data
		arguments: ax = matplotlib axis
				   title = str, plot title
				   bg_subtract = bool, if True, background subtract the data
				   scale = str, "linear" or "log" for colour scale
				   clip_interval = astropy.units.Quantity sets vmin and vmax for plot. e.g. Quantity([1,99]*u.percent)
				   
		"""
		if not ax:
			fig, ax = plt.subplots()
		data = copy.deepcopy(self.data)
		if self.frange is None:
			if self.obs_mode == 357:
				#assuming I-LOFAR mode 357 subband allocation
				#possible off-by-one error
				mask = np.zeros(data.shape)
				mask[200:256,:] = 1	#1st 200 sbs are mode 3, 56 blank sbs
				mask[456:512,:] = 1 #Next 200 sbs are mode 5, 56 blank sbs. Last 88 sbs are mode 7
				data = np.ma.array(data, mask=mask)
		if bg_subtract:
			data = (data / self.bg(data))
		imshow_args = {}
		date_format = dates.DateFormatter("%H:%M:%S")
		if scale == "log":
			data = np.log10(data)
		if clip_interval is not None:
			if len(clip_interval) == 2:
				clip_percentages = clip_interval.to('%').value
				vmin, vmax = AsymmetricPercentileInterval(*clip_percentages).get_limits(data)
			else:
				raise ValueError("Clip percentile interval must be specified as two numbers.")
			imshow_args["vmin"] = vmin
			imshow_args["vmax"] = vmax

		if self.trange is None:
			if len(file_split := self.bst_file.split("_bst")) > 1:
				ret = ax.imshow(data, aspect="auto", 
							extent=[self.times.plot_date[0], self.times.plot_date[-1], self.freqs.value[-1], self.freqs.value[0]],
							**imshow_args)
			else:
				ret = ax.imshow(data, aspect="auto", 
							extent=[self.obs_start.plot_date, self.obs_end.plot_date, self.freqs.value[-1], self.freqs.value[0]],
							**imshow_args)

		else:
			ret = ax.imshow(data, aspect="auto", 
						extent=[self.trange.start.plot_date, self.trange.end.plot_date, self.freqs.value[-1], self.freqs.value[0]],
						**imshow_args)			
		ax.xaxis_date()
		ax.xaxis.set_major_formatter(date_format)
		if title:
			ax.set_title(title)
		else:
			ax.set_title("I-LOFAR Mode {} Solar Observation".format(self.obs_mode))
		ax.set_xlabel("Start Time " + self.obs_start.iso[:-4] + "UTC")
		ax.set_ylabel("Frequency (MHz)")
		return ret

class LOFAR_BST_357(LOFAR_BST):
	"""
	Class for reading and plotting mode 357 observations from I-LOFAR
	This class uses the following assumptions:
	357 mode is always done with 488 subbands
	The subband numbers are defined as
	Mode 3: 54 -> 452 in steps of 2
	Mode 5: 54 -> 452 in steps of 2
	Mode 7: 54 -> 288 in steps of 2
	The integration time is always 1 second
	The target source is always the sun

	inputs: 
		bst_file = filename
		trange = Optional: time range (sunpy.time.TimeRange)
		frange = Optional: frequency range (2 element tuple, astropy.units.Quantity e.g [10,90]*u.MHz)
	"""
	def __init__(self, bst_file, trange=None, frange=None, integration=1):
		"""
		Initialise super with dummy sbs, obs_mode and frange arguments
		Insert spacing between modes as masked region in data
		"""
		self.bst_file = bst_file
		self.trange = trange
		self.frange = frange
		super().__init__(self.bst_file, sbs=np.arange(488), obs_mode=3, trange=self.trange, frange=None, bit_mode=8, integration=integration)
		self.frange = frange
		self.__get_data()#_frequency_correction()
	def __get_data(self):
	#def _frequency_correction(self):
		"""
		set up data array so that gaps in subbands are masked
		"""	
		sbs = np.array((np.arange(54,454,2),np.arange(54,454,2),np.arange(54,230,2)))
		blank_sbs = np.array((np.arange(454,512,2),np.arange(0,54,2),np.arange(454,512,2),np.arange(0,54,2)))
		obs_mode = np.array((3,5,7))
		blank_obs_mode = np.array((3,5,5,7))
		freqs = np.array([sb_to_freq(sb,mode) for sb,mode in zip(sbs, obs_mode)])
		blank_freqs = np.array([sb_to_freq(sb,mode) for sb,mode in zip(blank_sbs, blank_obs_mode)])
		
		self.sbs=np.concatenate((sbs[0], blank_sbs[0], blank_sbs[1], sbs[1],
									blank_sbs[2], blank_sbs[3], sbs[2]))

		self.freqs=np.concatenate((freqs[0], blank_freqs[0], blank_freqs[1], freqs[1],
									blank_freqs[2], blank_freqs[3], freqs[2]))
		
		
		blank_data = np.zeros((self.freqs.shape[0],self.data.shape[1]))
		#1st 200 sbs mode 3, blank, next 200 sbs mode 5, blank, last 88 sbs mode 7  
		blank_data[:200,:] = self.data[:200,:]
		blank_len_0 = len(blank_freqs[0]) + len(blank_freqs[1])
		blank_data[200 + blank_len_0:400 + blank_len_0,:] =self.data[200:400,:]
		blank_len_1 = len(blank_freqs[2]) + len(blank_freqs[3])
		blank_data[400 + blank_len_0 + blank_len_1 :,:] =self.data[400:,:]
		self.data = blank_data
		self.obs_mode = 357

		if self.frange is not None:
			f_start_end = []
			for freq in self.frange:
				diff_arr = np.abs(freq.to(u.MHz)-self.freqs)
				min_val = np.min(diff_arr)
				f_start_end.append(np.where(diff_arr == min_val)[0][0])
			f_start, f_end = f_start_end
		else:
			f_start = 0
			f_end = None
		self.freqs = self.freqs[f_start:f_end]
		self.data = self.data[f_start:f_end,:]		

class LOFAR_Raw(LOFAR_BST):
	"""
	Class to read and plot I-LOFAR raw data recorded with REALTA
	Required arguments: raw_file = File name,
	                                        sbs = subbands observed,
	                                        obs_mode = observation mode
	Optional arguments: trange = time range (sunpy.time.TimeRange), 
	                                        frange = frequency range (2 element list/array with start and end frequency as astropy.units.Quantity e.g [10,90]*u.MHz),
	                                        bit_mode = bit mode (usually 4, 8 or 16),
	                                        dtype = data type in raw file. Usually 32bit float for Stokes I
	                                        dt = time between data samples (seconds)

	Default values of bit_mode and dt are typical of I-LOFAR REALTA observations
	"""
	def __init__(self, raw_file, sbs, obs_mode, trange=None, frange=None, bit_mode=8, dtype=np.float32, dt=5.12e-6):
		self.raw_file = raw_file
		self.sbs = sbs
		self.obs_mode = obs_mode
		self.trange = trange
		self.frange = frange
		self.bit_mode = bit_mode
		self.dtype= dtype
		self.dt = TimeDelta(dt, format="sec")
		self.obs_start = Time(raw_file.split("_")[2])
		self.__get_data()
		self.obs_end = self.obs_start + (self.dt*self.data.shape[1])
	def __get_data(self):
		if self.trange is not None:
			dtype_ratio = {np.float32:4, np.int8:1}
			offset_time = self.trange.start - self.obs_start 
			offset = np.floor(offset_time/self.dt) * len(self.sbs) * dtype_ratio[self.dtype]
			offset = int(offset)
			count = np.floor(self.trange.seconds.value/self.dt.sec) * len(self.sbs)
			count = int(count)
			print(self.raw_file, self.dtype, count, offset)
			data = np.memmap(self.raw_file, self.dtype, shape = count, offset=offset, mode="c").reshape(-1,self.sbs.shape[0])
			
			
		else:
			data = np.memmap(self.raw_file, self.dtype, mode="c").reshape(-1,self.sbs.shape[0])
		
		self.freqs = sb_to_freq(self.sbs, self.obs_mode)

		if self.frange is not None:
			f_start_end = []
			for freq in self.frange:
				diff_arr = np.abs(freq.to(u.MHz)-self.freqs)
				min_val = np.min(diff_arr)
				f_start_end.append(np.where(diff_arr == min_val)[0][0])
			f_start, f_end = f_start_end
		else:
			f_start = 0
			f_end = None
		self.freqs = self.freqs[f_start:f_end]

		
		data = data[:,f_start:f_end].T
		self.data = np.flip(data, axis=0)

	def resample(self, new_dt):
		sum_int = int(np.floor(new_dt/self.dt.sec)) #how many indices long is new time step
		sum_shape = int(np.round(self.data.shape[1]/sum_int)) #determine array length
		sum_data = np.array([np.sum(self.data[:,i*sum_int:(i+1)*sum_int], axis=1) for i in range(sum_shape)])
		return sum_data

class LOFAR_Raw_357(LOFAR_Raw):
	"""
	Initialise super class with dummy variables
	"""
	def __init__(self, raw_file, trange=None, frange=None, dtype=np.float32, dt=5.12e-6):
		super().__init__(raw_file, sbs=np.arange(488), obs_mode=3, trange=trange, frange=None, bit_mode=8, dt=dt)
		self.frange = frange
		self.__get_data()

	def __get_data(self):
	
		"""
		set up data array so that gaps in subbands are masked
		"""	
		self.sbs = np.array((np.arange(54,454,2),np.arange(54,454,2),np.arange(54,230,2)))
		blank_sbs = np.array((np.arange(454,512,2),np.arange(0,54,2),np.arange(454,512,2),np.arange(0,54,2)))
		obs_mode = np.array((3,5,7))
		# blank_obs_mode = np.array((3,5,5,7))
		self.freqs = np.array([sb_to_freq(sb,mode) for sb,mode in zip(self.sbs, obs_mode)])
		# blank_freqs = np.array([sb_to_freq(sb,mode) for sb,mode in zip(blank_sbs, blank_obs_mode)])
		
		# self.sbs=np.concatenate((sbs[0], blank_sbs[0], blank_sbs[1], sbs[1],
		# 							blank_sbs[2], blank_sbs[3], sbs[2]))

		# self.freqs=np.concatenate((freqs[0], blank_freqs[0], blank_freqs[1], freqs[1],
		# 							blank_freqs[2], blank_freqs[3], freqs[2]))
		
		
		#blank_data = np.zeros((self.freqs.shape[0],self.data.shape[1]))
		#1st 200 sbs mode 3, blank, next 200 sbs mode 5, blank, last 88 sbs mode 7  
		#blank_data[:200,:] = self.data[:200,:]
		#blank_len_0 = len(blank_freqs[0]) + len(blank_freqs[1])
		#blanks_0 = np.zeros((blank_len_0, self.data.shape[1]))
		#blank_data[200 + blank_len_0:400 + blank_len_0,:] = self.data[200:400,:]
		#blank_len_1 = len(blank_freqs[2]) + len(blank_freqs[3])
		#blanks_1 = np.zeros((blank_len_1, self.data.shape[1]))
		#blank_data[400 + blank_len_0 + blank_len_1 :,:] = self.data[400:,:]
		#blank_data = np.insert(self.data, [199], blanks_0, axis=0)
		#blank_data = np.insert(blank_data, [200+blank_len_0+199], blanks_0, axis=0)
		#self.data = blank_data
		self.obs_mode = 357

		if self.frange is not None:
			f_start_end = []
			for freq in self.frange:
				diff_arr = np.abs(freq.to(u.MHz)-self.freqs)
				min_val = np.min(diff_arr)
				f_start_end.append(np.where(diff_arr == min_val)[0][0])
			f_start, f_end = f_start_end
		else:
			f_start = 0
			f_end = None
		self.freqs = self.freqs[f_start:f_end]
		self.data = self.data[f_start:f_end,:]		


# class LOFAR_From_Data(LOFAR_BST):
# 	def __init__(self, data, sbs, obs_mode, obs_type, trange=None, frange=None):
# 		self.data = data
# 		self.sbs = sbs
# 		self.freqs
