# from asyncore import file_dispatcher
# from lzma import FILTER_LZMA2
# from socket import IPX_TYPE
import lofar_data_class as ldc
import aoflagger
import sigpyproc as spp
import numpy as np 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from astropy.time import Time
from sunpy.time import TimeRange
import astropy.units as u
import datetime as dt
import logging

import matplotlib.dates as mdates

def run_aoflagger(stokes_list, filepath, fname, header):
    """
    Function takes 2d arrays of stokes parameters from observation and calculates the RFI flags
    """
    for key,stokes in stokes_list.items():

        #stokes.do_normalise()
        split=6
        arrs = np.array_split(stokes.data, split, axis=1)
        ratios = []

        time = Time(header.tstart, format = "mjd").isot
        time_len = header.nsamples
        time_res = header.tsamp
        trange = TimeRange(time, time_len*time_res*u.second, format='isot')
        tarrs = trange.split(split)

        frange = np.linspace(header.fbottom, header.ftop, hdr.nchans)

        for ix,j in enumerate(arrs):

            ntimes = j.shape[1]
            nch = j.shape[0]
            empties = np.zeros(j.shape)

            flagger = aoflagger.AOFlagger()
            #path = flagger.find_strategy_file(aoflagger.TelescopeId.Generic)
            strategy = flagger.load_strategy_file('/home/ofionnad/data/20211130_uranus/lofar-lba-wideband.lua')
            #The lua file above sets all of the inputs for the flagging algorithm. Need to edit that (or make a new one) to change thresholds etc.
            data = flagger.make_image_set(ntimes, nch, 2)

            data.set_image_buffer(0, j)
            data.set_image_buffer(1, empties)
            #data.set_image_buffer(1, empties)
            print("{}: Making image set for {}".format(dt.datetime.now().isoformat(), filepath+'/'+fname))

            flags = strategy.run(data)
            print("{}: Retrieving flags from buffer".format(dt.datetime.now().isoformat()))
            flagvalues = flags.get_buffer()
            ratio = float(sum(sum(flagvalues))) / (nch*ntimes)
            ratios.append(ratio)

            print("{}: Saving flags to binary file...".format(dt.datetime.now().isoformat()))
            np.save(filepath+'flags/'+str(key)+'_'+str(ix)+'.npy', flagvalues) #save binary pickle of flags

            print("{}: Flags shape:".format(dt.datetime.now().isoformat()) + str(flagvalues.shape))

            #normalising data
            print("{}: Normalising data".format(dt.datetime.now().isoformat()))
            norm_data = spp.Filterbank.FilterbankBlock(j, header).normalise()

            print("{}: Masking data and plotting 2 minutes from split data".format(dt.datetime.now().isoformat()))
            masked_arr = np.ma.masked_array(norm_data, mask=np.array(flagvalues).astype(int))
            #note just plot first 48828 times (~120 seconds)
            plotlen = 48828
            times = list(map(dt.datetime.fromisoformat, [tarrs[ix].start.value, str(tarrs[ix].start+plotlen*header.tsamp*u.second)]))
            times = mdates.date2num(times)
            f, ax = plt.subplots(figsize=(8,6)) 
            ax.imshow(masked_arr[:,:plotlen], origin='upper', aspect='auto', vmin=np.percentile(masked_arr, 5), vmax=np.percentile(masked_arr, 95), 
                        extent=[times[0], times[1], frange[0], frange[-1]])

            ax.xaxis_date()
            date_format = mdates.DateFormatter('%H:%M:%S.%f')
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_minor_locator(mdates.MinuteLocator())
            f.autofmt_xdate()

            plt.title('RFI masked stokes parameters {} - 2 minutes'.format(key))
            plt.xlabel('Time of observation from {}'.format(tarrs[ix].start))
            plt.tight_layout()
            plt.savefig(filepath+'flags/flagged_plot_{}{}.png'.format(key, ix), dpi=120)
            plt.close()
        
        print("{}: Percentage flags on data: ".format(dt.datetime.now().isoformat()) + str((sum(ratios)/split) * 100.0 ) + "%" )



if __name__=="__main__":

    fp = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/FR606/'
    fp2 = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/SE607/'
    fp3 = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/IE613/'
    for filepath in [fp, fp2, fp3]:
        print("{}: Running for observation: {}".format(dt.datetime.now().isoformat(), filepath))
        timefile = [f for f in listdir(filepath) if isfile(join(filepath, f)) and 'tstart' in f]
        with open(filepath+timefile[0], 'r') as f:
            tstart = f.read()
        tstart = tstart.split()[2]
        # filename = 'Uranus_FR606_2021-12-07T22:00:01.000_16chan_stokesI.fil'
        # f2 = filename.replace('_stokesI.fil', '') + '_stokesQ.fil'
        # f3 = filename.replace('_stokesI.fil', '') + '_stokesU.fil'
        # f4 = filename.replace('_stokesI.fil', '') + '_stokesV.fil'
        filenames = [f for f in listdir(filepath) if isfile(join(filepath, f)) and 'I.fil' in f]

        #read in the data as filterbank files in stokes form
        # ie = ldc.LofarFilterBank(filepath+filename, stokes=True, sbs=np.arange(51,461), obs_mode=3, tstart= tstart, tlimits=['20211130230100','20211130233000'])
        #I = ldc.LofarFilterBank(filepath+filename, stokes=True, sbs=np.arange(40,405), obs_mode=3, tstart= tstart)
        #Q = ldc.LofarFilterBank(filepath+f2, stokes=True, sbs=np.arange(40,405), obs_mode=3, tstart= tstart)
        #U = ldc.LofarFilterBank(filepath+f3, stokes=True, sbs=np.arange(40,405), obs_mode=3, tstart= tstart)
        #V = ldc.LofarFilterBank(filepath+f4, stokes=True, sbs=np.arange(40,405), obs_mode=3, tstart= tstart)

        stokes_list = [f.split('.fil')[0][-1] for f in filenames]
        # if 'FR606' in filepath:
        sbs = np.arange(40,405) # for FR station data
        # elif 'SE607' in filepath:
        #     sbs = np.arange(51,405) # for SE station data
        
        offset = 347 #size of header in bits of output filterbank files (can check this with sppReader.header.hdrlen )

        # for i in range(4):
        #     stokes_list[i].do_normalise()

        try:

            for ix,i in enumerate(filenames):
                #stokes_param = ldc.LofarFilterBank(filepath+i, stokes=True, sbs=np.arange(40,405), obs_mode=3, tstart=tstart) # i, q, u, v
                print("{}: Running stokes parameter {}".format(dt.datetime.now().isoformat(), stokes_list[ix]) )
                stokes_param = np.memmap(filepath+i, offset=offset, dtype='float32', mode='c').reshape(-1, (16*(sbs.shape[0]+1))).T
                hdr = spp.FilReader(filepath+i).header
                print(stokes_param.shape)
                run_aoflagger({stokes_list[ix]:stokes_param}, filepath, i, hdr)
                del stokes_param
                print("---")

        except Exception as e:
            print(dt.datetime.now().isoformat() + " - ERROR - : " + str(e))

        print("\n")


        #To downsample an array this works, example:
        # down = stokesI.reshape(-1, 16, stokesI.shape[1]).sum(1)
        # It adds another dimension temporarily and then sums along that dimension.
        # Takes data along channels in chunks of 16 and then sums them.
