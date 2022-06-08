import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nenupy.undysputed import Dynspec
from astropy.time import Time 
import astropy.units as u 
from os import path
from glob import glob
import numpy as np

def large_plot(fpath, time_range):
    fs = glob(path.join(fpath, '*spectra'))
    ds = Dynspec(lanefiles=fs)

    ds.time_range = time_range
    ds.bp_correction = 'standard'
    ds.jump_correction = True
    ds.rebin_dt = 0.12 * u.s 
    ds.rebin_df = 195.3125 * u.kHz

    #make list of data [oni, onv, off1i, off1v, off2i, off2v, etc]
    data = []
    for i in ds.beams:
        ds.beam = i
        data.append(ds.get(stokes='i'))
        data.append(ds.get(stokes='v'))

    tlims = mdates.date2num(Time(ds.time_range, format='unix').datetime)
    flims = [ds.fmin.value, ds.fmax.value]

    f, axd = plt.figure(constrained_layout=True).subplot_mosaic(
    """
    AB
    CD
    EF
    GH
    """, sharex=True, sharey=True,)

    for i,j in enumerate(axd):
        if data[i].polar[0]=='v':
            cm = "RdBu"
        else:
            cm = "viridis"
        j.imshow(data[i].amp.T, origin='lower', aspect='auto', cmap=cm, vmin=np.nanpercentile(data[i].amp[data[i].amp!=0], 5), 
              vmax=np.nanpercentile(data[i].amp[data[i].ampata!=0], 95), extent=[tlims[0], tlims[1], flims[0], flims[1]])
        if j==np.any(["B, D, F, H"]):
            j.set_yticklabels([])
        if j==np.any(["A", "B", "C", "D", "E", "F"]):
            j.set_xticklabels([])
        j.xaxis_date()
        
    axd.autofmt_xdate()
    plt.tight_layout()
