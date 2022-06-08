import aoflagger
import numpy as np 
import matplotlib.pyplot as plt
from astropy.time import Time
from sunpy.time import TimeRange
import astropy.units as u
import datetime as dt
from nenupy.undysputed import Dynspec
from glob import glob
from os.path import join

print('Set filename')
obs_path = '/mnt/ucc4_data3/data/Uranus/NenuFAR/20211130_230000_20211201_000000_URANUS_TRACKING'
dynspec_files = glob(join(obs_path, '*.spectra'))
ds = Dynspec(lanefiles=dynspec_files)

ds.bp_correction = 'standard' #bandpass correction
ds.jump_correction = True #remove gain jumps from analogue repointings (? I think) every 6 minutes

print('Getting stokes i from nenufar')
result = ds.get(stokes='i')

print('split data along t for flagging')
arrs = np.array_split(result.amp.T, 20, axis=1)
ratios = []
trange = TimeRange(ds.tmin, ds.tmax, format='unix')
tarrs = trange.split(20)
frange = np.linspace(ds.fmin.value, ds.fmax.value, result.amp.shape[1])
all_flags = []
for ix,j in enumerate(arrs):
    ntimes = j.shape[1]
    nch = j.shape[0]
    empties = np.zeros(j.shape)
    flagger = aoflagger.AOFlagger()
    strategy = flagger.load_strategy_file('/home/ofionnad/data/20211130_uranus/lofar-lba-wideband.lua')
    data = flagger.make_image_set(ntimes, nch, 2)
    data.set_image_buffer(0, j)
    data.set_image_buffer(1, empties)
    flags = strategy.run(data)
    flagvalues = flags.get_buffer()
    ratio = float(sum(sum(flagvalues)))/(nch*ntimes)
    ratios.append(ratio)
    all_flags.append(flagvalues)
    print('{} flagged'.format(ix))
    #np.save('/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_11_30/nenufar/flags/I_{}.npy'.format(ix), flagvalues)
all_flags = np.array(all_flags)
print('copy result for applying flags and downsampling')
out = np.copy(result)
out[all_flags==1] = np.nan
print('normalise')
out = out / np.nanmean(out, axis=1).reshape(out.shape[0], 1) #normalise 
print('downsample')
out_lowres = out.reshape(-1, 16, out.shape[1]).sum(1)
out_cumsum = np.cumsum(out_lowres, axis=1)
out_lowres = out_cumsum[:,12::12] - out_cumsum[:,:-12:12]
print('saving low res file with flags')
np.save('/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_11_30/nenufar/flagged_data.npy', out_lowres)
