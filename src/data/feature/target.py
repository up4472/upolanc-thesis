from anndata import AnnData
from pandas  import DataFrame
from types   import FunctionType
from typing  import Dict
from typing  import List
from typing  import Tuple

from tqdm.notebook import tqdm

import matplotlib
import seaborn

def extract_tpm_value (data : AnnData, groups : List[str], functions : List[FunctionType], layer : str = None) -> Tuple[Dict[str, Dict], Dict[str, List]] :
	"""
	Doc
	"""

	labels = dict()
	order  = dict()

	if layer is not None :
		matrix = data.layers[layer]
	else :
		matrix = data.X

	samples = data.obs
	genes   = data.var

	for gene in tqdm(genes.index) :
		labels[gene] = dict()

		cols = genes.index == gene

		for group in groups :
			tpkm = dict()

			for name in samples[group].unique() :
				rows = samples[group] == name

				vector = matrix[rows, cols]
				vector = vector.flatten()
				vector = [function(vector) for function in functions]

				tpkm[name] = vector

			group = group.lower()
			group_order = sorted(tpkm.keys())

			if group not in order.keys() :
				order[group] = group_order

			for index, function in enumerate(functions) :
				vals = [tpkm[key][index] for key in group_order]
				name = group + '-' + function.__name__

				labels[gene][name] = vals

	return labels, order

def extract_tpm_level (data : Dict[str, Dict], bounds : List[Tuple[str, float]], return_names : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	labels = dict()
	bounds = sorted(bounds, key = lambda x : x[1])

	for gene, table in tqdm(data.items()) :
		labels[gene] = dict()

		for key, value in table.items() :
			if return_names :
				array = ['unknown'] * len(value)
			else :
				array = [-1] * len(value)

			for index, (name, threshold) in enumerate(bounds, start = 0) :
				indices = [x >= threshold for x in value]
				current = name if return_names else index

				array = [current if flag else prev for prev, flag in zip(array, indices)]

			labels[gene][key] = array

	return labels

def distribution_group (data : Dict[str, Dict], genes : List[str], order : Dict[str, List], select : str = 'mean') -> Dict[str, Dict] :
	"""
	Doc
	"""

	dist = dict()
	gene = list(data.keys())[0]

	for group, items in data[gene].items() :
		if not group.endswith(select) :
			continue

		for index in range(len(items)) :
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

def distribution_histplot (data : Dict[str, Dict], groupby : str, discrete : bool = False, filename : str = None) -> None :
	"""
	Doc
	"""

	for gkey, group in data.items() :
		if groupby != gkey :
			continue

		n = len(group)

		match n :
			case  1 : nrows, ncols, gridlike = 1, 1, False
			case  2 : nrows, ncols, gridlike = 1, 2, False
			case  3 : nrows, ncols, gridlike = 1, 3, False
			case  4 : nrows, ncols, gridlike = 2, 2, True
			case  5 : nrows, ncols, gridlike = 2, 3, True
			case  6 : nrows, ncols, gridlike = 2, 3, True
			case  7 : nrows, ncols, gridlike = 2, 4, True
			case  8 : nrows, ncols, gridlike = 2, 4, True
			case  9 : nrows, ncols, gridlike = 3, 3, True
			case 10 : nrows, ncols, gridlike = 3, 4, True
			case 11 : nrows, ncols, gridlike = 3, 4, True
			case 12 : nrows, ncols, gridlike = 3, 4, True
			case 13 : nrows, ncols, gridlike = 4, 4, True
			case 14 : nrows, ncols, gridlike = 4, 4, True
			case 15 : nrows, ncols, gridlike = 4, 4, True
			case 16 : nrows, ncols, gridlike = 4, 4, True
			case 17 : nrows, ncols, gridlike = 5, 4, True
			case 18 : nrows, ncols, gridlike = 5, 4, True
			case 19 : nrows, ncols, gridlike = 5, 4, True
			case 20 : nrows, ncols, gridlike = 5, 4, True
			case _  : nrows, ncols, gridlike = n, 1, False

		kwargs = {
			'sharex'  : True,
			'sharey'  : True,
			'figsize' : (16, 10)
		}

		if gridlike :
			_, ax = matplotlib.pyplot.subplots(nrows, ncols, **kwargs)
		else :
			_, ax = matplotlib.pyplot.subplots(ncols, **kwargs)

		for index, (tkey, array) in enumerate(group.items()) :
			dataframe = DataFrame.from_dict({'Data' : array})
			axis = ax[index // ncols, index % ncols] if gridlike else ax[index]

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

		for index in range(len(group), nrows * ncols) :
			axis = ax[index // ncols, index % ncols] if gridlike else ax[index]
			axis.axis('off')

		if filename is not None :
			matplotlib.pyplot.savefig(
				filename + '.png',
				dpi    = 120,
				format = 'png'
			)
