import numpy as np 
import sigpyproc as spp 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import astropy.units as u 
from astropy.time import Time
from sunpy.time import TimeRange
from nenupy.undysputed import Dynspec
from nenufarData2 import plotter
import sys
from glob import glob
from os.path import join
from rich.pretty import pprint
import datetime
from scipy.stats import median_abs_deviation as mad
from seek.mitigation import sum_threshold


dirname = sys.argv[1]
outname = sys.argv[2]

# INPUT CONSTANTS
F1 = 16 # in mhz
F2 = 33
WINDOW1 = 0
WINDOW2 = 55 # minutes
CHUNK = 200000
BP = True # bandpass correction 
JUMP = True

#path example: '/mnt/ucc4_data3/data/Uranus/NenuFAR/20211214_220000_20211214_230000_URANUS_TRACKING/'
fs = glob(join(dirname, '*.spectra'))
ds = Dynspec(lanefiles=fs)

ds.freq_range = [F1*u.MHz, F2*u.MHz]
t1 = ds.tmin.datetime+datetime.timedelta(minutes=WINDOW1)
ds.time_range = [t1.isoformat(), (t1 + datetime.timedelta(minutes=WINDOW2)).isoformat()]

if BP:
    ds.bp_correction = 'median'
if JUMP:
    ds.jump_correction = True

ds.beam = 0
ds.rebin_dt = 120 * u.ms
on_i = ds.get(stokes='i')

time_list = [t1 - datetime.timedelta(seconds=0.12*x) for x in range(0, on_i.shape[0])]

mad1 = mad(on_i.amp).mean()
print('MAD 1:', mad1)

plt.figure()
plt.hist(on_i.amp.flatten(), bins=50)
plt.savefig('/home/ofionnad/data/uranus/2021_12_14/nenufar_histogram.png', dpi=120)
plt.close()

# normalise the frequency channel again
rowsum = on_i.amp.sum(axis=0)
on = (on_i.amp / rowsum[np.newaxis, :])

mad2 = mad(on)
print('MAD 2:', mad2.mean())

#print(on_i.amp.shape, on.shape)

f, ax = plt.subplots(2,2, figsize=(8,6))
ax[0,0].pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value, on[:CHUNK,:].T, vmin=np.nanpercentile(on, 5), vmax=np.nanpercentile(on, 95))
#ax[1].pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value, on[:CHUNK,:].T, vmin=np.nanpercentile(on, 5), vmax=np.nanpercentile(on, 95))


on_sn = np.copy(on).T
s1 = np.copy(on_sn).astype(float)
s2 = np.zeros_like(s1).astype(float)
print(on_sn.shape)
count = 0
#while np.any(np.abs(s1-s2) > 1e-5):

thresh = 1e-5
sdiff = np.abs(s1.std(axis=1)-s2.std(axis=1))
print(sdiff.shape)
while np.any(sdiff > thresh):
    count = 0
    s1 = on_sn.std(axis=1)
    #print(s1.shape)
    for rowi,row in enumerate(on_sn):
        if sdiff[rowi] > thresh:
            std_row = np.std(row)
            m = np.mean(row)
            for coli,data in enumerate(row):
                if data > (m+3*std_row):
                    on_sn[rowi,coli] = m
                    count += 1
                elif data < (m-3*std_row):
                    on_sn[rowi,coli] = m
                    count += 1
        else:
            pass
    print(count)
    s2 = on_sn.std(axis=1)
    #print(s2.shape)
    sdiff = np.abs(s1-s2)

#remove any signals > 4 sigma like in the paper
#on_sn[on_sn > on_sn.mean()+4*on_sn.std()] = on_sn.mean()

#sum every 16 channels to get back to default channel resolution
on_sn = on_sn.reshape(-1,16,on_sn.shape[-1]).sum(1)

mad3 = mad(on_sn)
print('MAD 3:', mad3)

ax[1,0].pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::16], on_sn[:,:CHUNK], vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95))
ax[0,1].hist(on_i.amp[:CHUNK,:].flatten(), bins=30)
ax[1,1].hist(on_sn[:,:CHUNK].flatten(), bins=30)

plt.tight_layout()
plt.savefig(outname, dpi=350)
plt.close()

plt.figure()
plt.pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::16], on_sn[:,:CHUNK], vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95))
plt.savefig(outname.strip('.png')+'_data'+'.png', dpi=350)
plt.close()


plt.pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::16], on_i.amp[:CHUNK,:].T[::16] - on_sn[:,:CHUNK], vmin=np.nanpercentile(on_i.amp[:CHUNK,:], 5), vmax=np.nanpercentile(on_i.amp[:CHUNK,:], 95))
plt.tight_layout()
plt.savefig(outname.strip('.png')+'_diff_'+'.png', dpi=350)
plt.close()

plt.pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::16], on_sn[:,:CHUNK], norm=colors.LogNorm(vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95)))
plt.tight_layout()
plt.savefig(outname.strip('.png')+'_out_'+'.png', dpi=350)
plt.close()

# snippet sums over different time resolutions and plots them:
#
# f, axs = plt.subplots(4)
# axs[0].pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::16], on_sn[:,:CHUNK], norm=colors.LogNorm(vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95)))
# on_sn = on_sn.reshape(-1,3,on_sn.shape[-1]).sum(1)
# norm_band = on_sn.sum(axis=1)
# on_sn = on_sn / norm_band[:,None]
# axs[1].pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::48], on_sn[:,:CHUNK], norm=colors.LogNorm(vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95)))
# on_sn = on_sn.reshape(-1,2,on_sn.shape[-1]).sum(1)
# axs[2].pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::96], on_sn[:,:CHUNK], norm=colors.LogNorm(vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95)))
# #on_sn = on_sn.reshape(-1,2,on_sn.shape[-1]).sum(1)
# on_sn = on_sn[3:,:].reshape(-1,2,on_sn.shape[-1]).sum(1)
# f = on_i.freq.to(u.MHz).value[::96][3:][::2]
# axs[3].pcolormesh(time_list[:CHUNK], f, on_sn[:,:CHUNK], norm=colors.LogNorm(vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95)))
# plt.tight_layout()
# plt.savefig(outname.strip('.png')+'_reslook_'+'.png', dpi=350)
# plt.close()

ress = [5, 9, 18, 36]
f, axs = plt.subplots(len(ress), figsize=(8,12))
print(on_sn.shape)
on_sn = on_sn[:-6,:]
#on_sn[:10,1500:1520] = 1e8 #injection


for i,j in enumerate(ress):
    on_f_lowres = on_sn.reshape(-1,j,on_sn.shape[-1]).sum(1)
    print(str(j),':\t', on_f_lowres.shape)

    data_fil = np.zeros_like(on_f_lowres)
    data_fil[on_f_lowres > on_f_lowres.mean()+4*on_f_lowres.std()] = 1

    for n,m in enumerate(data_fil):
        axs[i].plot(time_list[:CHUNK], m[:CHUNK]+n)
    
    data_and = np.sum(data_fil, axis=0)
    data_and[data_and == 1] = 0 # make sure there is more than 1 channel signal in each trigger
    
    axs[i].plot(time_list[:CHUNK], data_and[:CHUNK]+data_fil.shape[0], c='k')

plt.tight_layout()
plt.savefig(outname.strip('.png')+'_bit.png')
plt.close()

on_sn /= np.max(np.abs(on_sn),axis=0)
on_sn *= (255.0/on_sn.max())

plt.figure()
plt.pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value[::16][:-6], on_sn[:,:CHUNK], norm=colors.LogNorm(vmin=np.nanpercentile(on_sn[:,:CHUNK], 5), vmax=np.nanpercentile(on_sn[:,:CHUNK], 95)))
plt.tight_layout()
plt.savefig(outname.strip('.png')+'_singlebit.png', dpi=350)
plt.close()



### sum threshold version
rfi_mask = sum_threshold.get_rfi_mask(tod=np.ma.array(on), chi_1=0.00005, sm_kwargs=sum_threshold.get_sm_kwargs(40, 20, 15, 7.5),
                                di_kwargs=sum_threshold.get_di_kwrags(3, 7), plotting=False)
st_data = np.ma.array(np.flip(on, axis=0), mask=np.flip(rfi_mask, axis=0))
print(st_data.shape)

plt.pcolormesh(time_list[:CHUNK], on_i.freq.to(u.MHz).value, st_data[:CHUNK,:].T, vmin=np.nanpercentile(st_data, 5), vmax=np.nanpercentile(st_data, 95))
plt.tight_layout()
plt.savefig(outname.strip('.png')+'_sumthreshold.png', dpi=350)
plt.close()


# redo utr analysis with sumthresholded data
st_data = st_data.T
st_data = st_data.reshape(-1,16,st_data.shape[-1]).sum(1)

ress = [5, 9, 18, 36]
f, axs = plt.subplots(len(ress), figsize=(8,12))
st_data = st_data[:-6,:]


for i,j in enumerate(ress):
    on_f_lowres = st_data.reshape(-1,j,st_data.shape[-1]).sum(1)
    print(str(j),':\t', on_f_lowres.shape)

    data_fil = np.zeros_like(on_f_lowres)
    data_fil[on_f_lowres > on_f_lowres.mean()+4*on_f_lowres.std()] = 1

    for n,m in enumerate(data_fil):
        axs[i].plot(time_list[:CHUNK], m[:CHUNK]+n)
    
    data_and = np.sum(data_fil, axis=0)
    data_and[data_and == 1] = 0 # make sure there is more than 1 channel signal in each trigger
    
    axs[i].plot(time_list[:CHUNK], data_and[:CHUNK]+data_fil.shape[0], c='k')

plt.tight_layout()
plt.savefig(outname.strip('.png')+'_st_bit.png')
plt.close()