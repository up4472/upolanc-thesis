from anndata import AnnData
from pandas  import DataFrame
from typing  import Callable
from typing  import Dict
from typing  import List
from typing  import Tuple

from tqdm.notebook import tqdm

import itertools
import math
import matplotlib
import numpy
import seaborn

from src.data.feature._processing import boxcox1p_inv
from src.data.feature._processing import log1p_inv
from src.data.feature._processing import normalize_inv

def merge_dictionary (source : Dict, target : Dict) -> Dict :
	"""
	Doc
	"""

	for key, value in source.items() :
		if isinstance(value, dict) :
			merge_dictionary(
				source = value,
				target = target.setdefault(key, dict())
			)
		else :
			target[key] = value

	return target

def extract_tpm_single (data : AnnData, group : str, function : Callable, name : str, layer : str = None) -> Tuple[Dict, List] :
	"""
	Doc
	"""

	if layer is not None :
		matrix = data.layers[layer]
	else :
		matrix = data.X

	genes   = data.var
	samples = data.obs

	order = sorted(samples[group].unique())

	dataframe = DataFrame(
		data    = numpy.zeros((len(genes), len(order))),
		columns = order,
		index   = genes.index
	)

	for item in order :
		rows = samples[group] == item
		data = matrix[rows, :]

		dataframe[item] = function(x = data)

	name = group.lower() + '-' + name

	data = dataframe.transpose().to_dict(orient = 'list')
	data = {
		key : {name : item}
		for key, item in data.items()
	}

	return data, order

def extract_tpm_multi (data : AnnData, groups : List[str], functions : List[Tuple[str, Callable]], layer : str = None) -> Tuple[Dict, Dict] :
	"""
	Doc
	"""

	values = dict()
	order  = dict()

	for group, function in itertools.product(groups, functions) :
		result = extract_tpm_single(
			data     = data,
			layer    = layer,
			group    = group,
			function = function[1],
			name     = function[0]
		)

		values = merge_dictionary(
			source = result[0],
			target = values
		)

		order[group.lower()] = result[1]

	return values, order

def compute_percentile_bounds (data : Dict[str, Dict], group : str, classes : int) -> List[Tuple[str, float, float]]:
	"""
	Doc
	"""

	matrix = numpy.array([
		numpy.array(value[group])
		for value in data.values()
	])

	bounds = list()
	margin = 100 / classes

	for index in range(classes) :
		source = margin * index
		target = margin + source

		source = numpy.percentile(matrix, source)
		target = numpy.percentile(matrix, target)

		bounds.append((
			f'level-{index}',
			source,
			target
		))

	return bounds

def classify_tpm (data : Dict[str, Dict], classes : int = 5) -> Tuple[Dict[str, Dict], Dict[str, List]] :
	"""
	Doc
	"""

	labels = dict()
	bounds = {
		key : compute_percentile_bounds(
			data    = data,
			group   = key,
			classes = classes
		)
		for key in data[list(data.keys())[0]].keys()
		if not key.endswith('std')
	}

	for gene, table in tqdm(data.items()) :
		labels[gene] = dict()

		for key, value in table.items() :
			if key not in bounds.keys() :
				continue

			array = [numpy.nan] * len(value)

			for index, (name, low, high) in enumerate(bounds[key], start = 0) :
				indices = [x >= low for x in value]

				array = [
					index
					if flag else prev
					for prev, flag in zip(array, indices)
				]

			labels[gene][key] = array

	return labels, bounds

def display_bounds_mapping (bounds : Dict[str, List], group : str, min_value : float, max_value : float, box_lambda : float) -> None :
	"""
	Doc
	"""

	min_value  = 0
	max_value  = 4.700396577575031
	box_lambda = 0.276804378358019

	to_box = lambda x : normalize_inv(x,  min_value, max_value)
	to_log = lambda x : boxcox1p_inv(x, box_lambda)
	to_tpm = lambda x : log1p_inv(x, 2)

	bounds_norm = bounds[group]
	bounds_box  = [(name, to_box(x = low), to_box(x = high)) for name, low, high in bounds_norm]
	bounds_log  = [(name, to_log(x = low), to_log(x = high)) for name, low, high in bounds_box]
	bounds_tpm  = [(name, to_tpm(x = low), to_tpm(x = high)) for name, low, high in bounds_log]

	s32 = '-' * 32
	s21 = '-' * 21

	print('{:>31s} | {:>19s} | {:>19s} | {:>19s}'.format('TPM', 'log1p', 'boxcox1p', 'norm'))
	print('{}+{}+{}+{}'.format(s32, s21, s21, s21))

	format0 = lambda x : '{:9,.1f} - {:9,.1f}'.format(x[1], x[2])
	format1 = lambda x : '{:8.5f} - {:8.5f}'.format(x[1], x[2])

	for items in zip(bounds_tpm, bounds_log, bounds_box, bounds_norm) :
		print('{} : {} | {} | {} | {}'.format(
			items[0][0],
			format0(items[0]),
			format1(items[1]),
			format1(items[2]),
			format1(items[3]),
		))

def distribution_group (data : Dict[str, Dict], genes : List[str], order : Dict[str, List], select : str = 'mean') -> Dict[str, Dict] :
	"""
	Doc
	"""

	dist = dict()
	gene = list(data.keys())[0]

	groups = [
		(key, len(value))
		for key, value in data[gene].items()
		if key.endswith(select)
	]

	for group, length in groups :
		for index in range(length) :
			gkey = group.split('-')[0]
			tkey = order[gkey][index]

			if gkey not in dist.keys() :
				dist[gkey] = dict()
			if tkey not in dist[gkey].keys() :
				dist[gkey][tkey] = list()

			for gene in genes :
				dist[gkey][tkey].extend([
					data[gene][group][index]
				])

	return dist

def compute_gridsize (n : int) -> Tuple[int, int, int] :
	"""
	Doc
	"""

	if n == 1 : return 1, 1, 1
	if n == 2 : return 2, 1, 2
	if n == 3 : return 3, 1, 3

	nrows = math.ceil(math.sqrt(n))
	ncols = math.ceil(n / nrows)

	return n, nrows, ncols

def distribution_histplot (data : Dict[str, Dict], groupby : str, discrete : bool = False, filename : str = None) -> None :
	"""
	Doc
	"""

	for gkey, group in data.items() :
		if groupby != gkey :
			continue

		n, nrows, ncols = compute_gridsize(
			n = len(group)
		)

		kwargs = {
			'sharex'  : True,
			'sharey'  : True,
			'figsize' : (nrows * 16, ncols * 10)
		}

		if ncols > 1 :
			_, ax = matplotlib.pyplot.subplots(nrows, ncols, **kwargs)
		else :
			_, ax = matplotlib.pyplot.subplots(ncols, **kwargs)

		for index, (tkey, array) in enumerate(group.items()) :
			dataframe = DataFrame.from_dict({'Data' : array})

			if nrows == 1 or ncols == 1 :
				axis = ax[index]
			else :
				axis = ax[index // ncols, index % ncols]

			seaborn.histplot(
				x         = 'Data',
				data      = dataframe,
				ax        = axis,
				color     = '#799FCB',
				discrete  = discrete,
				alpha     = 0.9,
			)

			axis.set_title(tkey.title())
			axis.set_xlabel('')
			axis.set_ylabel('')

		for index in range(n, nrows * ncols) :
			if nrows == 1 or ncols == 1 :
				axis = ax[index]
			else :
				axis = ax[index // ncols, index % ncols]

			axis.axis('off')

		if filename is not None :
			matplotlib.pyplot.savefig(
				filename + '.png',
				dpi    = 120,
				format = 'png'
			)
