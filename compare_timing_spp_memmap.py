import numpy as np
from sigpyproc.readers import FilReader
# from sigpyproc.block import FilterbankBlock
# import dask

fr_file_data = '/mnt/ucc2_scratch/data/ofionnad/uranus/lofar_2021/2021_12_07/FR606/Uranus_FR606_2021-12-07T22:00:01.000_16chan_stokesI.fil'
fr = FilReader(fr_file_data)
fr_data = fr.read_block(100000, 150000)
fr_lowres = fr_data.downsample(10, 16)

fr_lowres2 = fr_data.reshape(-1, 16, fr_data.shape[1]).sum(1)