from pandas           import DataFrame
from torch            import Tensor
from torch.nn         import Conv1d
from torch.nn         import Conv2d
from torch.nn         import Module
from torch.nn         import Sequential
from torch.utils.data import DataLoader
from typing           import Any
from typing           import Dict
from typing           import List
from typing           import Union
from typing           import Tuple

import logomaker
import matplotlib
import numpy
import torch
import math
import scipy

from source.python.dataset.dataset_classes import SequenceDataset
from source.python.encoding.onehot         import onehot_encode

def get_conv_output_for_layer (layer : Module, sequence : numpy.ndarray, device : Any = None) -> Tensor :
	"""
	Doc
	"""

	sequence = torch.tensor(sequence).unsqueeze(0).permute(0, 2, 1)

	if device is not None :
		sequence = sequence.to(device)

	with torch.no_grad() :
		output = layer(sequence.double())

	return output[0]

def get_conv_layers_from_model (model : Module) -> Dict[str, List] :
	"""
	Doc
	"""

	elements = list(model.children())

	layers  = list()
	weights = list()
	biases  = list()

	for index, layer in enumerate(elements) :
		if isinstance(layer, (Conv1d, Conv2d)) :
			layers.append( layer)
			weights.append(layer.weight)
			biases.append( layer.bias)

		if isinstance(layer, (Module, Sequential)) :
			subdata = get_conv_layers_from_model(model = layer)

			layers.extend( subdata['layer'])
			weights.extend(subdata['weight'])
			biases.extend( subdata['bias'])

	return {
		'layer'  : layers,
		'weight' : weights,
		'bias'   : biases
	}

def get_kernel_activations (sequences : List[str], mapping : Dict[str, numpy.ndarray], layer : Module, device : Any, weighted : bool = False, threshold : Tuple[float, float] = None) -> Dict[int, Tensor] :
	"""
	Doc
	"""

	filters = layer.out_channels
	height  = len(mapping['A'])
	width   = layer.kernel_size[0]
	motifs  = numpy.zeros((filters, width, height), dtype = numpy.float64)

	print('Filters : {}'.format(motifs.shape[0]))
	print('Width   : {}'.format(motifs.shape[1]))
	print('Height  : {}'.format(motifs.shape[2]))
	print()

	dataset    = SequenceDataset(sequences)
	dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)

	l = width // 2
	r = width - l

	f1 = lambda x, l, r, y : x[l :r, :] * y	# noqa shadows : weight activation - range
	f2 = lambda x, l, r, y : x[l :r, :]		# noqa shadows : binary activation - yes or no

	softmax = lambda x : scipy.special.softmax(x, axis = 0)
	reduce  = lambda x : numpy.add.reduce(x)
	ones    = lambda   : numpy.ones((width, height), dtype = numpy.float64)

	if weighted : apply = f1
	else        : apply = f2

	layer = layer.to(device)

	for sequence in dataloader :
		sequence = sequence[0]

		matrix = onehot_encode(
			sequence  = sequence,
			mapping   = mapping,
			default   = None,
			transpose = False
		)

		output = get_conv_output_for_layer(
			layer    = layer,
			device   = device,
			sequence = matrix
		).detach().cpu().numpy()

		if threshold is None : condition = output
		else                 : condition = numpy.logical_and(threshold[0] < output, output < threshold[1])

		indices = numpy.nonzero(condition)

		idx = indices[0]
		pos = indices[1]

		activations = [
			softmax(
				reduce([
					apply(
						x = matrix,
						l = position - l,
						r = position + r,
						y = output[fid, position]
					)
					if position - l >= 0 and position + r <= len(sequence)
					else ones()
					for position in pos[idx == fid]
				])
				if len(pos[idx == fid]) > 0
				else ones()
			)
			for fid in range(filters)
		]

		motifs = numpy.add(motifs, activations, out = motifs)

	return motifs

def plot_kernels (weights : List[Tensor], nucleotide_order : Union[str, List] = None, figsize : Tuple[int, int] = None, filename : str = None, rows : int = 4, cols : int = 8) -> None :
	"""
	Doc
	"""

	if figsize is None : figsize = (16, 10)

	if nucleotide_order is None :
		nucleotide_order = 'ACGT'

		print('Using default nucleotide order : {}'.format(nucleotide_order))
		print()

	if isinstance(nucleotide_order, str) :
		nucleotide_order = [x for x in nucleotide_order]

	max_size = rows * cols
	sum_size = len(weights)

	num_plot = math.ceil(sum_size / max_size)

	kernels = [torch.nn.functional.softmax(x, dim = 0).cpu().detach().numpy() for x in weights]
	vmin = numpy.min(kernels)
	vmax = numpy.max(kernels)

	print('Minimum Value : {:.5f}'.format(vmin))
	print('Maximum Value : {:.5f}'.format(vmax))
	print()

	for plotid in range(num_plot) :
		a = max_size * (0 + plotid)
		b = max_size * (1 + plotid)

		data = kernels[a:b]

		filters = len(data)
		height  = data[0].shape[0]
		width   = data[0].shape[1]
		rows    = int(numpy.ceil(filters / cols))

		figure, ax = matplotlib.pyplot.subplots(
			nrows   = rows,
			ncols   = cols,
			sharex  = True,
			sharey  = True,
			figsize = figsize
		)

		ax = ax.flatten()

		x = numpy.arange(width)
		y = numpy.arange(height)

		for index, kernel in enumerate(data) :
			corrected_index = int(index + plotid * max_size)

			img = ax[index].imshow(kernel, vmin = vmin, vmax = vmax, cmap = 'gray')

			ax[index].set_yticks(y)
			ax[index].set_yticklabels(nucleotide_order)
			ax[index].set_xticks(x)
			ax[index].set_title('Filter {}'.format(corrected_index))
			ax[index].grid(visible = False)

		figure.subplots_adjust(right = 0.985)
		figure.colorbar(
			mappable = img, # noqa
			cax      = figure.add_axes([0.990, 0.15, 0.01, 0.70])
		)

		matplotlib.pyplot.grid(False)

		if filename is not None :
			matplotlib.pyplot.savefig(
				filename + '-{}.png'.format(plotid),
				dpi         = 120,
				format      = 'png',
				bbox_inches = 'tight'
			)

		if plotid != 0 :
			matplotlib.pyplot.close(figure)

def plot_kernels_and_motifs (weights : List[Tensor], activations : Dict[int, Tensor], to_type : str, nucleotide_order : Union[str, List] = None, figsize : Tuple[int, int] = None, filename : str = None, rows : int = 4, cols : int = 8) -> None :
	"""
	Doc
	"""

	if figsize is None : figsize = (16, 10)

	if nucleotide_order is None :
		nucleotide_order = 'ACGT'

		print('Using default nucleotide order : {}'.format(nucleotide_order))
		print()

	if isinstance(nucleotide_order, str) :
		nucleotide_order = [x for x in nucleotide_order]

	max_size = rows * cols
	sum_size = len(weights)

	num_plot = math.ceil(sum_size / max_size)

	kernels = [torch.nn.functional.softmax(x, dim = 0).cpu().detach().numpy() for x in weights]
	vmin = numpy.min(kernels)
	vmax = numpy.max(kernels)

	print('Minimum Value : {:.5f}'.format(vmin))
	print('Maximum Value : {:.5f}'.format(vmax))
	print()

	for plotid in range(num_plot) :
		a = max_size * (0 + plotid)
		b = max_size * (1 + plotid)

		data = kernels[a:b]

		filters = len(data)
		height  = data[0].shape[0]
		width   = data[0].shape[1]
		rows    = int(numpy.ceil(filters / cols)) * 2

		figure, ax = matplotlib.pyplot.subplots(
			nrows   = rows,
			ncols   = cols,
			sharex  = True,
			sharey  = False,
			figsize = figsize
		)

		ax = ax.flatten()

		x = numpy.arange(width)
		y = numpy.arange(height)

		for index, kernel in enumerate(data) :
			corrected_index = int(index + plotid * max_size)

			row = index // cols
			col = index  % cols

			kindex = (2 * row + 0) * cols + col
			lindex = (2 * row + 1) * cols + col

			img = ax[kindex].imshow(kernel, vmin = vmin, vmax = vmax, cmap = 'gray')

			ax[kindex].set_yticks(y)
			ax[kindex].set_yticklabels(nucleotide_order)
			ax[kindex].set_xticks(x)
			ax[kindex].set_title('Filter {}'.format(corrected_index))
			ax[kindex].grid(visible = False)

			df = DataFrame(activations[corrected_index], columns = nucleotide_order)
			df = logomaker.transform_matrix(df, from_type = 'counts', to_type = to_type)

			logomaker.Logo(df, ax = ax[lindex])

			ax[lindex].grid(visible = False)
			ax[lindex].set_xticks([])
			ax[lindex].set_yticks([])

		figure.subplots_adjust(right = 0.985)
		figure.colorbar(
			mappable = img, # noqa
			cax      = figure.add_axes([0.990, 0.15, 0.01, 0.70])
		)

		matplotlib.pyplot.grid(False)

		if filename is not None :
			matplotlib.pyplot.savefig(
				filename + '-{}.png'.format(plotid),
				dpi         = 120,
				format      = 'png',
				bbox_inches = 'tight'
			)

		if plotid != 0 :
			matplotlib.pyplot.close(figure)
