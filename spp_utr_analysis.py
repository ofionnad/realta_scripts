import numpy as np 
import sigpyproc as spp 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import astropy.units as u 
from astropy.time import Time
from sunpy.time import TimeRange
from nenufarData2 import plotter
import sys
from rich.pretty import pprint

fname = sys.argv[1]
f1 = 25 #these are array channel numbers, not mhz
f2 = 60

#fname = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_28/nenufar/heimdall/heimdall_32bit_fullres_time_2021-12-28T20:07:41.196_2021-12-28T20:14:10.392_beam0.fil'
plotname = fname.split('nenufar')[0]+'nenufar/heimdall/'+ 'bitrep_plot_'+fname.split('_')[-2]
plotname2 = fname.split('nenufar')[0]+'nenufar/heimdall/'+ 'downres_data_plot_'+fname.split('_')[-2]+'.png'

fr = spp.FilReader(fname)
pprint(fr.header)
data = fr.readBlock(0, fr.header.nsamples)
#need to get rid of nans
ts = fr.header.obs_date.split('/')
t = TimeRange('-'.join(ts[::-1])+'T'+fr.header.obs_time, fr.header.tobs*u.s, format='isot')
tlims = mdates.date2num([t.start.datetime, t.end.datetime])
flims = [fr.header.ftop, fr.header.fbottom]
data = np.nan_to_num(data, nan=np.nanmean(data))

#downsample the data
data_down = data.downsample(tfactor=64, ffactor=32)[f1:f2,:] #tfactor=64 gives a tres of ~150ms, ffactor of 16 gives a fres of ~96kHz
print(data_down.shape)
data_down = data_down.normalise()
mean = data_down.mean()
std = data_down.std()
data_down[data_down > mean + 4*std] = mean
data_down[data_down < mean - 4*std] = mean

bitrep = np.array([np.where(j>j.mean()+3*j.std(), 1, 0) for j in data_down])
bitsum = bitrep.sum(axis=0)

#only add the bit if it is consecutive with another in frequency direction
bitsum2 = np.zeros(bitrep.shape[1])
for i,j in enumerate(bitrep):
    if i==0:
        pass
    bitsum2 += (np.logical_and(j, bitrep[i-1]))

f, ax = plt.subplots(2, sharex=True)
ax[0].imshow(data_down, origin='lower', aspect='auto', vmin=np.nanpercentile(data_down[data_down!=0], 5), 
            vmax=np.nanpercentile(data_down[data_down!=0], 95), extent=[tlims[0], tlims[1], flims[0], flims[1]])
ax[1].plot(np.linspace(tlims[0], tlims[1], bitsum.shape[0]), bitsum)
ax[1].plot(np.linspace(tlims[0], tlims[1], bitsum2.shape[0]), bitsum2)

ax[1].xaxis_date()
f.autofmt_xdate()
plt.tight_layout()
plt.grid(which='major')
plt.savefig(plotname+'.png', dpi=150)

plt.close()

#plotter(data_down, tlims, flims, plotname2)