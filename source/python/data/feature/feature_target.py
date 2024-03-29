from anndata               import AnnData
from pandas                import DataFrame
from sklearn.preprocessing import LabelBinarizer
from typing                import Any
from typing                import Callable
from typing                import Dict
from typing                import List
from typing                import Tuple

import itertools
import math
import matplotlib
import numpy
import seaborn

from source.python.data.feature.feature_processing import boxcox1p_inv
from source.python.data.feature.feature_processing import log1p_inv
from source.python.data.feature.feature_processing import normalize_inv
from source.python.data.stats.stats_statistics     import interquartile_range
from source.python.data.stats.stats_statistics     import zscore

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

def extract_tpm_single (data : AnnData, group : str, function : Callable, name : str, layer : str = None, outlier_filter : str = None, outlier_params : Dict[str, float] = None, verbose : bool = False) -> Tuple[Dict, List] :
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

		y = function(x = data, axis = 0)
		p = 1.0

		if outlier_filter is not None :
			if outlier_filter == 'iqr' :
				data, _, _, p = interquartile_range(
					data = data,
					axis = 0,
					k    = outlier_params['factor-iqr'],
					n    = 1
				)

			if outlier_filter == 'zscore' :
				data, _, _, p = zscore(
					data = data,
					axis = 0,
					ddof = 1,
					z    = outlier_params['factor-zscore'],
					n    = 1
				)

		x = function(x = data, axis = 0)

		if outlier_filter is not None :
			ymean = numpy.nanmean(y)
			xmean = numpy.nanmean(x)

			p = 100.0 * (1.0 - p)
			p = numpy.nanmean(p)
			d = 100.0 * (abs(ymean - xmean) / ymean)

			if verbose :
				print('Filtered out [{:8.4f} %] percent with mean change [{:8.4f} %] from [{}] [{}] [{}]'.format(p, d, group, name, item))

		dataframe[item] = x

	name = group.lower() + '-' + name

	data = dataframe.transpose().to_dict(orient = 'list')
	data = {
		key : {name : item}
		for key, item in data.items()
	}

	return data, order

def extract_tpm_multi (data : AnnData, groups : List[str], functions : List[Tuple[str, Callable]], layer : str = None, outlier_filter : str = None, outlier_params : Dict[str, float] = None, verbose : bool = False) -> Tuple[Dict, Dict] :
	"""
	Doc
	"""

	values = dict()
	order  = dict()

	for group, function in itertools.product(groups, functions) :
		result = extract_tpm_single(
			data           = data,
			layer          = layer,
			group          = group,
			function       = function[1],
			name           = function[0],
			outlier_filter = outlier_filter,
			outlier_params = outlier_params,
			verbose        = verbose
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

	matrix = matrix.flatten()
	matrix = matrix[~numpy.isnan(matrix)]

	bounds = list()
	margin = 100 / classes

	for index in range(classes) :
		source = margin * index
		target = margin + source

		source = max(0.0, min(100.0, source))
		target = max(0.0, min(100.0, target))

		source = numpy.percentile(matrix, source, axis = None)
		target = numpy.percentile(matrix, target, axis = None)

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

	for gene, table in data.items() :
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

def display_bounds_mapping (bounds : List[Tuple[str, float, float]], start : str, values : Dict[str, Any], mapping : Dict[str, str]) -> None :
	"""
	Doc
	"""

	min_value  = values['min_value']
	max_value  = values['max_value']
	box_lambda = values['box_lambda']
	log_base   = values['log_base']

	def compute_inverse (item : List[Tuple[str, float, float]], src : str) -> List[Tuple[str, Any, Any]] :
		if   src == 'boxcox1p' : func = lambda x : boxcox1p_inv(x = x, lmbda = box_lambda)
		elif src == 'log1p'    : func = lambda x : log1p_inv(x = x, base = log_base)
		elif src == 'normal'   : func = lambda x : normalize_inv(x = x, min_value = min_value, max_value = max_value)
		else : raise ValueError()

		return [(x, func(low), func(high)) for x, low, high in item]

	source = start
	bounds = [bounds]
	names  = [source]

	while source is not None :
		bounds.append(compute_inverse(
			item = bounds[-1],
			src  = source
		))

		source = mapping[source]
		names.append(source if source is not None else 'tpm')

	print(' ' * 8, end = '')
	print(' | '.join('{:>29s}'.format(name) for name in names))

	print('-' * 8 + '-' * (29 * len(names) + 3 * (len(names) - 1)))

	for data in zip(*bounds) :
		print('{:8s}'.format(data[0][0]), end = '')
		print(' | '.join(['{:13,.5f} - {:13,.5f}'.format(item[1], item[2]) for item in data]))

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

	if nrows > ncols :
		return n, ncols, nrows

	return n, nrows, ncols

def distribution_histplot (data : List[Dict], names : List[str], groupby : str, discrete : bool = False, title : bool = False, filename : str = None) -> None :
	"""
	Doc
	"""

	for gkey, group in data[0].items() :
		if groupby != gkey :
			continue

		n, nrows, ncols = compute_gridsize(
			n = len(group)
		)

		kwargs = {
			'sharex'  : True,
			'sharey'  : True,
			'figsize' : (ncols * 16, nrows * 10)
		}

		if ncols > 1 : fig, ax = matplotlib.pyplot.subplots(nrows, ncols, **kwargs)
		else         : fig, ax = matplotlib.pyplot.subplots(       ncols, **kwargs)

		fig.tight_layout()

		for index, tkey in enumerate(group.keys()) :
			if   nrows == 1 and ncols == 1 : axis = ax
			elif nrows == 1  or ncols == 1 : axis = ax[index]
			else                           : axis = ax[index // ncols, index % ncols]

			dictionary = {
				'Value' : list(),
				'Group' : list()
			}

			for i, x in enumerate(data) :
				values = x[gkey][tkey]
				groups = [names[i] for _ in values]

				dictionary['Value'].extend(values)
				dictionary['Group'].extend(groups)

			seaborn.histplot(
				x        = 'Value',
				hue      = 'Group',
				data     = DataFrame.from_dict(dictionary),
				ax       = axis,
				discrete = discrete,
				alpha    = 0.65
			)

			if title :
				axis.set_title(tkey.title())

			axis.set_xlabel('')
			axis.set_ylabel('')

		for index in range(n, nrows * ncols) :
			if nrows == 1 or ncols == 1 : axis = ax[index]
			else                        : axis = ax[index // ncols, index % ncols]

			axis.axis('off')

		if filename is not None :
			matplotlib.pyplot.savefig(
				filename + '.png',
				dpi         = 120,
				format      = 'png',
				bbox_inches = 'tight',
				pad_inches  = 0
			)

def create_mapping (values : Dict[str, Any], labels : Dict[str, Any], order : Dict[str, Any]) -> Tuple[Dict, Dict, Dict] :
	"""
	Doc
	"""

	features_binarizer = dict()
	features_grouped   = dict()
	features_exploded  = dict()

	transcripts = list(values.keys())
	functions   = ['mean', 'max']
	groups      = ['tissue', 'age', 'group', 'perturbation', 'global']

	for group, function in itertools.product(groups, functions) :
		group_order = order[group]

		dataframe = DataFrame()
		binarizer = LabelBinarizer()
		binarizer = binarizer.fit(group_order)

		key = f'{group}-{function}'
		group = group.capitalize()

		dataframe['ID']         = transcripts
		dataframe['Transcript'] = transcripts
		dataframe['TPM_Value']  = [values[x][key] for x in transcripts]
		dataframe['TPM_Label']  = [labels[x][key] for x in transcripts]
		dataframe[group]        = [group_order    for _ in transcripts]

		features_grouped[key] = dataframe

		dataframe = dataframe.copy().explode(['TPM_Value', 'TPM_Label', group])
		dataframe = dataframe.reset_index(drop = True)

		dataframe['TPM_Label'] = dataframe['TPM_Label'].astype('category')
		dataframe[group] = dataframe[group].astype('category')

		features_binarizer[key] = binarizer
		features_exploded[key] = dataframe

	return features_binarizer, features_grouped, features_exploded
