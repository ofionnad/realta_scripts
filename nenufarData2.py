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
from os.path import join, isfile, isdir
from os import listdir 
import argparse
import logging
#from skimage.measure import block_reduce
#from sigpyproc.io.fileio import FileWriter #use for version 1.0.0 but out files cannot be read by digifil!
import sigpyproc as spp

def plotter(data, tlims, flims, fname):
    #data[data==0] = np.nan
    f, ax = plt.subplots(figsize=(8,6))
    p1 = ax.imshow(data, origin='lower', aspect='auto', vmin=np.nanpercentile(data[data!=0], 5), 
              vmax=np.nanpercentile(data[data!=0], 95), extent=[tlims[0], tlims[1], flims[0], flims[1]])
    ax.xaxis_date()
    f.autofmt_xdate()
    f.colorbar(p1)
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()


def get_parser():
    parser = argparse.ArgumentParser(description='Handles nenufar data (.spectra) and analyses the data for UEDs')
    parser.add_argument("-i", "--input", dest="fp", help="Path to nenufar .spectra files")
    parser.add_argument("-o", "--output", dest="out_fp", help="Output directory for files")
    parser.add_argument("-f", "--flag", dest="flag", help="Run AOflagger or not. Default: False")
    parser.add_argument("-b", "-beam", dest="beam", help="Which beam to use from nenufar data. 0 1 2 or 3")
    parser.add_argument("-nsplit", dest="nsplit", help="Number of arrays to chunk for aoflagger")
    parser.add_argument("-s", "--stokes", dest="stokes", help="Stokes parameter to calculte from data, i.e i or v")
    parser.add_argument("-tlimits", dest="tlimits", nargs="+", help="Time limits for nenufar data (isot format). [time1, time2]")
    parser.add_argument("-flimits", dest="flimits", nargs="+", help="Freqency limits for nenufar data (MHz). [freq1, freq2]")
    parser.add_argument("-tf", dest="tfactor", help="Time downsampling factor plotting and saving")
    parser.add_argument("-ff", dest="ffactor", help="Frequency downsampling factor to use (in kHz)")

    args = parser.parse_args()

    return args


def set_logger(args):
    logger = logging.getLogger('nenupy')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    fh = logging.FileHandler(args.out_fp+'{}_nenufarData.log'.format(dt.datetime.now().strftime("%y%m%d_%H%M%S")))
    logger.addHandler(fh)
    fh.setFormatter(formatter)


def times(args, ds):

    if args.tlimits:
        if len(args.tlimits)==1:
            #set one limit from command line and other from ds header
            end = Time(args.tlimits[0]).datetime
            start = ds.tmin.datetime
        elif len(args.tlimits)==2:
            start, end = Time(args.tlimits[0]).datetime, Time(args.tlimits[1]).datetime
        else:
            raise IndexError("-tlimits option must have only 1 or 2 times within the observation range")

    else:
        #read times from dynspec header
        start, end = ds.tmin.datetime, ds.tmax.datetime

    return TimeRange(start, end)


def read_nenuheader(args):
    logger = logging.getLogger('nenupy')
    fs = glob(join(args.fp, '*.spectra'))
    logger.info("Nenufar spectra lane files: {}".format([s.split('/')[-1] for s in fs]))
    #fs = sorted(fs, key=lambda x: x[-9])
    return Dynspec(lanefiles=fs)
    

def read_nenudata(args, ds, timerange):

    ds.bp_correction = 'standard'
    #logger = logging.getLogger('nenupy')

    if timerange.seconds > 6*u.min:
        ds.jump_correction = True

    ds.time_range = [timerange.start.isot, timerange.end.isot]

    if args.flimits:
        ds.freq_range = [float(args.flimits[0]), float(args.flimits[1])]

    data = ds.get(stokes=args.stokes).amp

    return data #nsamples x nchans

# def spp_filewriter(data, args, t, ds):
#     if metadata.metadata('sigpyproc')['Version']!='1.0.0':
#         raise Exception("Wrong sigpyproc version! Need to use version 1.0.0!")
#     filwriter = FileWriter(args.out_fp+'heimdall/heimdall_32bit_fullres_time_{}_{}_beam{}.fil'.format(t.start, t.end, ds.beam), mode='w', nbits=32, tsamp=ds.dt.value, nchans=data.shape[1])
#     filwriter.cwrite(data)
#     filwriter.close()

def concat_fil(path, fname, beam):
    files = [f for f in listdir(path) if isfile(join(path, f)) if '8bit' in f and 'fil' in f and 'beam'+str(beam) in f]
    files = np.sort(files)
    data = []
    nsamples = 0
    for f in files:
        fr = spp.FilReader(path+f)
        data.append(fr.readBlock(0, fr.header.nsamples))
        nsamples += fr.header.nsamples
    fr = spp.FilReader(path+files[0]) # first header with proper timestamp
    data = np.hstack(data)
    fr_concat = spp.Filterbank.FilterbankBlock(data, 
        spp.Header.Header(info={'tsamp':fr.header.tsamp, 'nsamples':nsamples, 'nchans':fr.header.nchans,
        'fch1':fr.header.fbottom+(np.abs(fr.header.foff)/2), 'foff':(fr.header.ftop-fr.header.fbottom)/fr.header.nchans, 
        'tstart':fr.header.tstart, 'ftop':fr.header.ftop, 'fbottom':fr.header.fbottom}))
    fr_concat.toFile(filename=path+fname, back_compatible=True)
    

def spp_filewriter_old(data, args, t, ds):
    print(data.shape)
    hdr = spp.Header.Header(info={'nbits':32, 'tsamp':ds.dt.value, 'tstart':t.start.mjd, 'nchans': data.shape[0], 'nsamples': data.shape[1], 'fch1':float(args.flimits[0]), 'foff':ds.df.to(u.MHz).value, 'ibeam':ds.beam})
    output = {}
    logger = logging.getLogger('nenupy')
    logger.info(args.out_fp + 'heimdall/heimdall_32bit_fullres_time_{}_{}_beam{}.fil'.format(t.start.isot, t.end.isot, ds.beam))
    output['data'] = hdr.prepOutfile(args.out_fp+'heimdall/heimdall_32bit_fullres_time_{}_{}_beam{}.fil'.format(t.start.isot, t.end.isot, ds.beam), updates = { 'nifs': 1 }, back_compatible = True)
    output['data'].cwrite(data.T.ravel())
    #f = spp.Utils.File(args.out_fp+'heimdall_32bit_fullres_time_{}_{}_beam{}.fil'.format(t.start, t.end, ds.beam), mode='w')
    #f.cwrite(data)
    """
    {'nbits': 32,
    'tsamp': 0.024,
    'nchans': 192,
    'src_raj': 0.0,
    'src_dej': 0.0,
    'nifs': 1,
    'hdrlen': 124,
    'filelen': 768124,
    'nbytes': 768000,
    'nsamples': 1000,
    'filename': 'test.fil',
    'basename': 'test',
    'extension': '.fil',
    'tobs': 24.0,
    'ra': '00:00:00.0000',
    'dec': '00:00:00.0000',
    'ra_rad': 0.0,
    'dec_rad': 0.0,
    'ra_deg': 0.0,
    'dec_deg': 0.0,
    'dtype': '<f4'}
    """

def run_aoflagger(ix, args, ds, t, data):

    flagger = aoflagger.AOFlagger()
    strat_file = '/home/ofionnad/data/20211130_uranus/nenufar-default.lua'
    strategy = flagger.load_strategy_file(strat_file)
    logger = logging.getLogger('nenupy')
    logger.info("Flagging strategy file loaded: {}".format(strat_file))

    logger.info("Running AOFlagger on data chunk #{}\n\t\t\tData Shape: {}\n\t\t\tData Max: {}\n\t\t\tData Min: {}\n\t\t\tStokes: {}\n".format(ix, data.shape, data.max(), data.min(), args.stokes))

    """
    Running flagger
    """
    data_buffer = flagger.make_image_set(*data.shape, 1)
    data_buffer.set_image_buffer(0, data.T) #nchans x nsamples
    logger.info('Data buffer set, running flagger...')
    flags = strategy.run(data_buffer)
    flagvalues = flags.get_buffer()
    ratio = float(sum(sum(flagvalues)))/(data.shape[0]*data.shape[1])
    logger.info("Flagged {:.4f}%% of data".format(ratio*100.0))

    datasave = np.copy(data.T)
    datasave[flagvalues==1] = np.nan
    if ix == 0:
        tlims = mdates.date2num([t.start.datetime, t.end.datetime])
        plotter(datasave, tlims, [float(args.flimits[0]), float(args.flimits[1])], args.out_fp+'dynspec_beam{}_time{}_{}.png'.format(ds.beam, t.start.isot, t.end.isot))    
    spp_filewriter_old(datasave, args, t, ds)


def main():

    args = get_parser()
    set_logger(args)
    #logger.info(args)

    ds = read_nenuheader(args)
    timerange = times(args, ds)
    tarrs = timerange.split(int(args.nsplit)) #split up the timerange we use at a time

    #pipeline
    for ix, t in enumerate(tarrs):
        for beam in range(2):
            ds.beam = beam # set different beams from data recording for each loop
            data = read_nenudata(args, ds, t)

            if args.flag:
                run_aoflagger(ix, args, ds, t, data)

            del data



if __name__=="__main__":

    try:
        main()
    except Exception as e:
        logger = logging.getLogger('nenupy')
        logger.exception("main crashed. Error: %s", e)