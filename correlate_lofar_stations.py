import numpy as np 
import dask.array as da
import matplotlib.pyplot as plt 
# import sigpyproc as spp 
import os
from numba import njit, types, prange, generated_jit
from sigpyproc.core import kernels
from sigpyproc.readers import FilReader
from sigpyproc.block import FilterbankBlock
from sigpyproc.io.sigproc import parse_header, edit_header

from scipy.signal import correlate2d
from astropy.time import Time
from sunpy.time import TimeRange
import astropy.units as u


def concatenate_flags(filepath, offset, shapes):

    files = list(np.sort([f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f)) and '.npy' in f]))
    all_flags = [ (np.memmap(filepath+f, dtype=np.uint8, mode='r', offset=offset, shape=shapes[ix] ) ) for ix, f in enumerate(files)]
    return da.concatenate(all_flags, axis=1)

def get_header(file):
    edit_header(file, "telescope_id", 0)
    edit_header(file, "machine_id", 0)
    return FilReader(file).header

def get_data(datafile, header, offset, nchans, limits):
    data = np.memmap(datafile, offset=offset, dtype=np.float32, mode='c').reshape(-1, (nchans)).T[:, limits[0]:limits[1]]
    return FilterbankBlock(data, header)

def make_masked(data, flags):
    return np.ma.masked_array(data, mask=flags)

def do_normalise(data):
    return data / (np.nanmean(data, axis=1).reshape(data.shape[0], 1))

def normalise_bandwidth(data):
    return data / (np.nanmean(data, axis=0).reshape(1, data.shape[1]))
"""
@njit(cache=True)
def downcast(intype, result):
    if isinstance(intype, types.Integer):
        return lambda intype, result: np.uint8(result)
    return lambda intype, result: np.float32(result)

@njit(cache=True, parallel=True, locals={"temp": types.f8})
def downsampling(data: np.memmap, tfactor: int=1, ffactor: int=1):
    '''
    Copy of sigpyproc downsample, works with memmaps
    '''
    
    if data.shape[0] % ffactor != 0:
        raise ValueError("Bad frequency factor!")
    
    ar = data.T.ravel().copy()
    nchans = data.shape[0]
    nsamps = data.shape[1]
    new_ar = np.empty((nchans*nsamps) // (tfactor*ffactor), dtype=ar.dtype)
    for isamp in prange(nsamps//tfactor):
        for ichan in range(nchans//ffactor):
            pos = nchans * isamp * tfactor + ichan *ffactor
            temp = 0
            for ifactor in range(tfactor):
                ipos = pos + ifactor*nchans
                for jfactor in range(ffactor):
                    temp += ar[ipos+jfactor]
            new_ar[(nchans//ffactor)*isamp + ichan] = downcast(ar[0], temp/(ffactor*tfactor))

    return new_ar
"""

def downsampling(data, tfactor, ffactor):

    if data.shape[0] % ffactor != 0:
        raise ValueError("Bad frequency factor!")

    ar = data.T.ravel().copy()
    new_ar = kernels.downsample_2d(ar, tfactor, ffactor, data.shape[0], data.shape[1])
    new_ar = new_ar.reshape(data.shape[1]//tfactor, data.shape[0]//ffactor).transpose()

    return new_ar

def do_downsample(data, tfactor, ffactor):

    if data.shape[0] % ffactor != 0:
        raise ValueError("Bad frequency factor!")

    newarr = data.reshape(-1, ffactor, data.shape[1]).sum(1)

    if tfactor != 1:
        newarr_sum = np.cumsum(newarr, axis=1)
        outarr = newarr_sum[:,tfactor::tfactor] - newarr_sum[:,:-tfactor:tfactor]
        return outarr
    else:
        return newarr

def prep_data(filename, flagname, flagshapes, tres = 0.03, offset=347, limits=[None, None]):
    fil = FilReader(filename)
    fil_header = fil.header
    fil_data = np.memmap(filename, offset=offset, dtype='float32', mode='c').reshape(-1, fil_header.nchans).T[:, limits[0]:limits[1]]
    flags = concatenate_flags(flagname, offset=128, shapes=flagshapes)
    fil_data[flags.astype(bool)] = 0
    fil_norm = do_normalise(fil_data)
    fil_norm[np.isnan(fil_norm)] = 0 #turn nan to 0, i think 0 works best for downsampling
    fil_lowres = do_downsample(fil_data, int(tres//fil_header.tsamp), 16)
    return fil_lowres, fil_header

def align_times(data1, data2, tres, header1, header2):
    tstart1, tstart2 = Time(header1.tstart, format='mjd'), Time(header2.tstart, format='mjd')
    tdiff = tstart2-tstart1
    nsamples = round(tdiff.sec / tres)
    if tdiff > 0:
        data1 = data1[:,nsamples:]
    elif tdiff < 0:
        data2 = data2[:,nsamples:]
    
    return data1, data2

if __name__=="__main__":

    tres = 0.03 #downsampled time resolution

    flagout = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/aoflagger_220126.out'
    with open(flagout) as f:
        flagshapes = [tuple(map(int, l.split(':')[-1].strip('\n()').split(','))) for l in f if 'Flags shape:' in l]

    #get FR606 data from 7th Dec 2021
    print('reading fr data')
    fr_file_data = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/FR606/Uranus_FR606_2021-12-07T22:00:01.000_16chan_stokesI.fil'
    edit_header(fr_file_data, "telescope_id", 0)
    edit_header(fr_file_data, "machine_id", 0)
    fr_file_flags = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/FR606/flags/'
    fr_lowres, fr_header = prep_data(fr_file_data, fr_file_flags, flagshapes[:6])

    print('reading se data')
    se_file_data = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/SE607/Uranus_SE607_2021-12-07T22:00:00.000_16chan_stokesI.fil'
    se_file_flags = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/SE607/flags/'
    edit_header(se_file_data, "telescope_id", 0)
    edit_header(se_file_data, "machine_id", 0)    
    se_lowres, se_header = prep_data(se_file_data, se_file_flags, flagshapes[6:12])

    fr_lowres, se_lowres = align_times(fr_lowres, se_lowres, tres, fr_header, se_header)

    f, axs = plt.subplots(2)
    axs[0].imshow(fr_lowres, origin='upper', aspect='auto', vmin=np.nanpercentile(fr_lowres, 5), vmax=np.nanpercentile(fr_lowres, 95))
    axs[1].imshow(se_lowres, origin='upper', aspect='auto', vmin=np.nanpercentile(se_lowres, 5), vmax=np.nanpercentile(se_lowres, 95))
    plt.tight_layout()
    plt.savefig('/home/ofionnad/data/uranus/2021_12_07/fr_se.png', dpi=180)
    plt.close()

    print("Cross-correlation 2d")
    #cross correlation
    option1 = correlate2d(fr_lowres, se_lowres, mode='full', boundary='fill', fillvalue=0)

    print("Cross correlation 1d")
    option2 = []
    for i, freq in enumerate(fr_lowres):
        option2.append(np.correlate(freq, se_lowres[i], mode='full'))

    option2 = np.array(option2)
    print("plot and compare")
    print(option1.shape, option2.shape)

    f, axs = plt.subplots(2)
    axs[0].imshow(option1, origin='upper', aspect='auto', vmin=np.nanpercentile(option1, 5), vmax=np.nanpercentile(option1, 95))
    axs[1].imshow(option2, origin='upper', aspect='auto', vmin=np.nanpercentile(option2, 5), vmax=np.nanpercentile(option2, 95))
    plt.tight_layout()
    plt.savefig('/home/ofionnad/data/uranus/2021_12_07/fr_se_correlation.png', dpi=120)
    plt.close()   
