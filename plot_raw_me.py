import sys
import astropy.units as u
from astropy.units import Quantity
from sunpy.time import TimeRange
import matplotlib.pyplot as plt
import numpy as np
path = "/mnt/ucc2_data1/data/ofionnad/20201013_jupiter"
sys.path.insert(0, path)
from lofar_raw import LOFAR_Raw
fname = sys.argv[1]
trange = TimeRange("2020-10-13 17:47:00", 10*u.min)
frange = [15,60]*u.MHz
sbs = np.arange(76,584)
mode = 3
raw = LOFAR_Raw(fname, sbs, mode, frange=frange, trange=trange)
title = "I-LOFAR - Jupiter - Stokes V"
fig, ax = pltsubplots(figsize=(10,8))
raw.plot(ax=ax, title=title, scale='linear', bg_subtract=True, clip_interval=Quantity([5,95]*u.percent))
plt.savefig(path+'Jupiter-Stokes-v.png')
plt.show()


