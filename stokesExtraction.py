import sigpyproc as spp
import argparse
import numpy as np

def extractData(sppReader, prefix = None, stokesI = True, stokesQ = False, stokesU = False, stokesV = True, gulp = 262144, **kwargs):
	output = {}
	assert(sppReader.header.nifs == 4)
	assert((gulp % 4) == 0)
	if stokesI:
		output['stokesI'] = sppReader.header.prepOutfile(prefix.replace('.fil', '') + "_stokesI.fil",
						updates = { 'nifs': 1 }, back_compatible = True)
	if stokesQ:
		output['stokesQ'] = sppReader.header.prepOutfile(prefix.replace('.fil', '') + "_stokesQ.fil",
						updates = { 'nifs': 1 }, back_compatible = True)
	if stokesU:
		output['stokesU'] = sppReader.header.prepOutfile(prefix.replace('.fil', '') + "_stokesU.fil",
						updates = { 'nifs': 1 }, back_compatible = True)
	if stokesV:
		output['stokesV'] = sppReader.header.prepOutfile(prefix.replace('.fil', '') + "_stokesV.fil",
						updates = { 'nifs': 1 }, back_compatible = True)


	for __, __, data in sppReader.readPlan(gulp, **kwargs):
		data = data.reshape(-1, sppReader.header.nchans).T
		if stokesI:
			output['stokesI'].cwrite((np.square(data[:, 0::4]) + np.square(data[:, 1::4])).T.ravel())

		if stokesQ:
			output['stokesQ'].cwrite((np.square(data[:, 0::4]) - np.square(data[:, 1::4])).T.ravel())

		if stokesU:
			output['stokesU'].cwrite((2 * data[:, 2::4]).T.ravel())

		if stokesV:
			output['stokesV'].cwrite((-2 * data[:, 3::4]).T.ravel())


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Convert a voltage correlation fitlerbank to Stokes parameter filterbanks")
	parser.add_argument('-i', dest = 'input', type = str, help = "Input file")
	parser.add_argument('-o', dest = 'prefix', type = str, default = '', help = "Output file prefix")
	parser.add_argument('-I', dest = 'I', action = 'store_true', default = False, help = 'Generate Stokes I output')
	parser.add_argument('-Q', dest = 'Q', action = 'store_true', default = False, help = 'Generate Stokes Q output')
	parser.add_argument('-U', dest = 'U', action = 'store_true', default = False, help = 'Generate Stokes U output')
	parser.add_argument('-V', dest = 'V', action = 'store_true', default = False, help = 'Generate Stokes V output')
	parser.add_argument('-s', dest = 'gulp', type = int, default = 262144, help = "Samples to read per iteration")
	args = parser.parse_args()

	if args.I == args.Q == args.U == args.V == False:
		raise RuntimeError("No work provided")
	sppReader = spp.FilReader(args.input)
	if len(args.prefix) == 0:
		args.prefix = args.input.rstrip('.fil')

	extractData(sppReader, args.prefix, args.I, args.Q, args.U, args.V, args.gulp * 4)
