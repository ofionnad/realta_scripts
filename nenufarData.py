import aoflagger
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
from sunpy.time import TimeRange
import astropy.units as u
import datetime as dt
from nenupy.undysputed import Dynspec
from glob import glob
from os.path import join, isdir
import argparse
import logging
#from skimage.measure import block_reduce
from numpy.lib.stride_tricks import as_strided
from sigpyproc.io.fileio import FileWriter


"""
1. Read in the nenufar data in a nice quick way
2. Flag the RFI using aoflagger 
3. Apply the flags, downsample the data, and normalise it
4. Cross-correlate with other stations
"""


def block_reduce(image, block_size=2, func=np.sum, cval=0, func_kwargs=None):
    """Copied from scikit-image module"""

    if np.isscalar(block_size):
        block_size = (block_size,) * image.ndim
    elif len(block_size) != image.ndim:
        raise ValueError("`block_size` must be a scalar or have "
                         "the same length as `image.shape`")

    if func_kwargs is None:
        func_kwargs = {}

    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = np.pad(image, pad_width=pad_width, mode='constant',
                   constant_values=cval)

    blocked = view_as_blocks(image, block_size)

    return func(blocked, axis=tuple(range(image.ndim, blocked.ndim)),
                **func_kwargs)


def view_as_blocks(arr_in, block_shape):

    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


def read_nenufar(fp:str, tlimits:list=None, flimits:list=None, beam:int=None) -> Dynspec:
    """
    Make nenupy dynspec object from undysputed module.
    By default corrects for flat bandpass and pointing jump correction.
    """
    fs = glob(join(fp, '*.spectra'))
    logger.info("Nenufar spectra lane files: {}".format([s.split('/')[-1] for s in fs]))
    #fs = sorted(fs, key=lambda x: x[-9])
    ds = Dynspec(lanefiles=fs)
    ds.bp_correction = 'standard'
    ds.jump_correction = False

    if tlimits:
        print(tlimits)
        logger.info("New time limits: {}".format(tlimits))
        ds.time_range = tlimits
        if (ds.time_range[1] - ds.time_range[0]) > 360:
            logger.info("Data longer than 6 minutes")
            ds.jump_correction = True
        else:
            ds.jump_correction=False
    else:
        ds.jump_correction = True

    if flimits: ds.freq_range = [flimits[0], flimits[1]]

    if beam: ds.beam = beam

    return ds

def get_stokes(ds: Dynspec, stokes) -> list:
    """
    Return array of stokes parameters from dynspec nenupy object
    """
    return ds.get(stokes=stokes)

def downsample_2d(data, ffactor, tfactor):
    if tfactor==1 and ffactor==1:
        return data 
    else:
        return block_reduce(data, block_size=(ffactor, tfactor), func=np.sum, cval=0)

def plotter(data, tlims, flims, fname):
    f, ax = plt.subplots(figsize=(8,6))
    p1 = ax.imshow(data, origin='lower', aspect='auto', vmin=np.nanpercentile(data, 5), 
              vmax=np.nanpercentile(data, 95), extent=[tlims[0], tlims[1], flims[0], flims[1]])
    ax.xaxis_date()
    f.autofmt_xdate()
    f.colorbar(p1)
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()

def run_aoflagger(ds: Dynspec, data, nsplit: int, outfp:str, beam, tfactor:int = 12, ffactor:int = 32, tlimits=None):
    flagger = aoflagger.AOFlagger()
    #strat_file = '/home/ofionnad/data/20211130_uranus/nenufar_strategy.lua'
    #strat_file = '/home/ofionnad/data/20211130_uranus/lofar-lba-wideband.lua'
    strat_file = '/home/ofionnad/data/20211130_uranus/nenufar-default.lua'
    strategy = flagger.load_strategy_file(strat_file)
    logger = logging.getLogger('nenupy')
    logger.info("Flagging strategy file loaded: {}".format(strat_file))

    arrs = np.array_split(data.amp.T, nsplit, axis=1)

    logger.info("Data:\n\t\tData Shape: {}\tMax Data: {}\tMin Data: {}\n\tNo. split arrays: {}\tSplit array shape: {}".format(data.amp.T.shape, data.amp.max(), data.amp.min(), len(arrs), arrs[0].shape))
    ratios = []
    if tlimits:
        trange = TimeRange(tlimits[0], tlimits[1], format='isot')
    else:
        trange = TimeRange(ds.time_range[0], ds.time_range[1], format='unix')
    tarrs = trange.split(nsplit)
    #frange = np.linspace(ds.fmin.value, ds.fmax.value, data.amp.T.shape[1])
    all_flags = []
    for ix,j in enumerate(arrs):
        ntimes = j.shape[1]
        nch = j.shape[0]
        #empties = np.zeros(j.shape)
        data_buf = flagger.make_image_set(*j.T.shape, 1)
        data_buf.set_image_buffer(0, j)
        #data_buf.set_image_buffer(1, empties)
        logger.info('Data buffer set, running flagger...')
        flags = strategy.run(data_buf)
        flagvalues = flags.get_buffer()
        ratio = float(sum(sum(flagvalues)))/(nch*ntimes)
        ratios.append(ratio)
        all_flags.append(flagvalues)
        #np.save('/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_11_30/nenufar/flags/I_{}.npy'.format(ix), flagvalues)
        if ix==0:
            logger.info("Plotting first data chunk")
            tlims = mdates.date2num([tarrs[ix].start.datetime, tarrs[ix].end.datetime])
            flims = [ds.fmin.value, ds.fmax.value]
            t = dt.datetime.now().strftime("%H%M%S")
            outname1 = outfp+'{}_stokes{}_beam{}_trange{}_{}.png'.format(str(t), str(data.polar[0]), str(beam), tarrs[ix].start, tarrs[ix].end)
            plotter(j, tlims, flims, outname1)
            # f, ax = plt.subplots(figsize=(8,6))
            # ax.imshow(j, origin='lower', aspect='auto', vmin=np.nanpercentile(j, 5), 
            #             vmax=np.nanpercentile(j, 95), extent=[tlims[0], tlims[1], ds.fmin.value, ds.fmax.value])
            # ax.xaxis_date()
            # f.autofmt_xdate()
            # plt.tight_layout()
            # plt.savefig(outfp+'{}_stokes{}_beam{}.png'.format(str(t), str(data.polar[0]), str(beam)), dpi=150)
            # plt.close()
            # f, ax = plt.subplots(figsize=(8,6))
            temp = np.copy(j)
            temp[flagvalues==1] = 0
            temp = temp / np.nanmean(temp, axis=1).reshape(temp.shape[0], 1)
            temp = np.ma.masked_where(temp==0, temp)
            outname2 = outfp+'{}_stokes{}_beam{}_trange{}_{}_flagged.png'.format(str(t), str(data.polar[0]), str(beam), tarrs[ix].start, tarrs[ix].end)
            plotter(temp, tlims, flims, outname2)
            # plt.imshow(temp, origin='lower', aspect='auto', vmin=np.nanpercentile(temp, 5), 
            #             vmax=np.nanpercentile(temp, 95), extent=[tlims[0], tlims[1], ds.fmin.value, ds.fmax.value])
            # ax.xaxis_date()
            # f.autofmt_xdate()
            # plt.tight_layout()
            # plt.savefig(outfp+'{}_stokes{}_beam{}_flagged.png'.format(str(t), str(data.polar[0]), str(beam)), dpi=150)

    logger.info("Flagging percentage: {}".format(str((sum(ratios)/nsplit) * 100.0)))
    all_flags = np.hstack(all_flags)
    out = np.copy(data.amp.T) # (nchans, nsamples)
    out[all_flags==1] = 0 #could use nan here either.
    filwriter = FileWriter('heimdall_32bit_fullres_time_{}_{}.fil'.format(trange.start, trange.end), mode='w', nbits=8*out.itemsize, tsamp=ds.dt.value, nchans=out.shape[1])
    filwriter.cwrite(out)
    filwriter.close()
    #out = out / np.nanmean(out, axis=1).reshape(out.shape[0], 1)
    #out_lowres = downsample_2d(out, int(ffactor), int(tfactor))
    #out_lowres = out_lowres / np.nanmean(out_lowres, axis=1).reshape(out_lowres.shape[0], 1)
    # out_lowres = out.reshape(-1, 32, out.shape[1]).sum(1)
    # out_cumsum = np.cumsum(out_lowres, axis=1)
    # out_lowres = out_cumsum[:,int(tres//ds.dt.value)::int(tres//ds.dt.value)] - out_cumsum[:,:int(-tres//ds.dt.value):int(tres//ds.dt.value)]
    #logger.info("Saving flagged, downsampled data in pickle format")
    #np.save(outfp+'data_flagged_lowres_beam{}.npy'.format(str(beam)), out_lowres)
    #tlims = mdates.date2num([ds.tmin.datetime, ds.tmax.datetime])
    #flims =  [ds.fmin.value, ds.fmax.value]
    #logger.info("Plotting the downsampled data after renormalisation")
    #out_lowres = np.ma.masked_where(out_lowres==0, out_lowres)
    #plotter(out_lowres, tlims, flims, outfp+'data_flagged_lowres_beam{}.png'.format(str(beam)))

def main(args, logger):

    logger.info("Nenufar datapath: {}".format(args.fp))
    for fp in args.fp:
        if not isdir(fp):
            logger.error("Directory does not exist: {}".format(fp))
            raise OSError("Directory does not exist")

        if args.beam:
            logger.info("Single beam: {}\nRunning pipeline once".format(args.beam))
            ds = read_nenufar(fp, args.tlimits, args.flimits, int(args.beam))
            data = get_stokes(ds, args.stokes)
            if args.flag:
                run_aoflagger(ds, data, int(args.nsplit), str(args.out_fp), int(args.beam), float(args.tfactor), float(args.ffactor))
        else:
            logger.info("All beams\nRunning pipeline for each beam.")
            for beam in range(4):
                logger.info("Beam #{}".format(beam))
                if args.tlimits:
                    ds = read_nenufar(fp, args.tlimits, args.flimits, beam)
                    data = get_stokes(ds, args.stokes)    
                    if args.flag:
                        run_aoflagger(ds, data, int(args.nsplit), str(args.out_fp), beam, float(args.tfactor), float(args.ffactor), args.tlimits)
                else:
                    trange = TimeRange(args.tlimits[0], args.tlimits[1], format='isot').split(10)
                    for ix,t in enumerate(trange):
                        ds = read_nenufar(fp, trange, args.flimits, beam)
                        data = get_stokes(ds, args.stokes)
                        if args.flag:
                            run_aoflagger(ds, data, int(args.nsplit), str(args.out_fp), beam, float(args.tfactor), float(args.ffactor), [t.start, t.end])

                del ds, data #clear up memory for next beam        


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Handles nenufar data (.spectra) and analyses the data for UEDs')
    parser.add_argument("-i", "--input", dest="fp", nargs='+', help="Path to nenufar .spectra files")
    parser.add_argument("-o", "--output", dest="out_fp", help="Output directory for files")
    parser.add_argument("-f", "--flag", dest="flag", help="Run AOflagger or not. Default: False")
    parser.add_argument("-b", "-beam", dest="beam", help="Which beam to use from nenufar data. 0 1 2 or 3")
    parser.add_argument("-nsplit", dest="nsplit", help="Number of arrays to chunk for aoflagger")
    parser.add_argument("-s", "--stokes", dest="stokes", help="Stokes parameter to calculte from data, i.e I or V")
    parser.add_argument("-tlimits", dest="tlimits", nargs="+", help="Time limits for nenufar data (isot format). [time1, time2]")
    parser.add_argument("-flimits", dest="flimits", nargs="+", help="Freqency limits for nenufar data (MHz). [freq1, freq2]")
    parser.add_argument("-tf", dest="tfactor", help="Time downsampling factor plotting and saving")
    parser.add_argument("-ff", dest="ffactor", help="Frequency downsampling factor to use (in kHz)")
    #parser.add_argument()
    args = parser.parse_args()

    logger = logging.getLogger('nenupy')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    fh = logging.FileHandler(args.out_fp+'{}_nenufarData.log'.format(dt.datetime.now().strftime("%y%m%d_%H%M%S")))
    logger.addHandler(fh)
    fh.setFormatter(formatter)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception("main crashed. Error: %s", e)
