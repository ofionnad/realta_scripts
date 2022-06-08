import sys
from glob import glob
from os.path import join
import datetime
import astropy.units as u 
from nenupy.undysputed import Dynspec

dirname = sys.argv[1]
outname = sys.argv[2]

# INPUT CONSTANTS
F1 = 15 # in mhz
F2 = 33
WINDOW1 = 0
WINDOW2 = 20 # minutes
#CHUNK = 12000
BP = True # bandpass correction 
JUMP = True #Change to false if data doesn't cover a gain jump (every 6 minutes)

#path example: '/mnt/ucc4_data3/data/Uranus/NenuFAR/20211214_220000_20211214_230000_URANUS_TRACKING/'
# nenufar obs from the 14th dec 2021.
fs = glob(join(dirname, '*.spectra'))
ds = Dynspec(lanefiles=fs)

ds.freq_range = [F1*u.MHz, F2*u.MHz]
t1 = ds.tmin.datetime+datetime.timedelta(minutes=WINDOW1)
ds.time_range = [t1.isoformat(), (t1 + datetime.timedelta(minutes=WINDOW2)).isoformat()]

if BP:
    ds.bp_correction = 'standard'
if JUMP:
    ds.jump_correction = True

ds.beam = 0
ds.rebin_dt = 120 * u.ms #make the time resolution 120 ms
on_i = ds.get(stokes='i')