from pandas import DataFrame
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib
import numpy
import pandas
import seaborn

def show (data : DataFrame, rows : int = 5, cols : int = 10) -> DataFrame :
	"""
	Doc
	"""

	print(f'Number of rows : {data.shape[0]}')
	print(f'Number of cols : {data.shape[1]}')

	if rows is None :
		rows = len(data)

	if len(data.columns) > cols :
		return data.iloc[:, :cols].head(n = rows)

	return data.head(n = rows)

def filter_samples (data : DataFrame, cutoff : Dict[str, Any] = None) -> Tuple[DataFrame, Dict[str, str]] :
	"""
	Doc
	"""

	columns = data.columns.tolist()

	dictionary = dict()
	max_cutoff = None
	sum_cutoff = None
	avg_cutoff = None
	std_cutoff = None
	px0_cutoff = None

	if cutoff is not None :
		if 'max' in cutoff.keys() and cutoff['max']    > 0.0 : max_cutoff = cutoff['max']
		if 'sum' in cutoff.keys() and cutoff['sum']    > 0.0 : sum_cutoff = cutoff['sum']
		if 'avg' in cutoff.keys() and cutoff['avg']    > 0.0 : avg_cutoff = cutoff['avg']
		if 'std' in cutoff.keys() and cutoff['std']    > 0.0 : std_cutoff = cutoff['std']
		if 'px0' in cutoff.keys() and cutoff['px0'][0] > 0.0 : px0_cutoff = cutoff['px0']

	for name in columns :
		column = data[name]

		if all(item == column[0] for item in column) :
			dictionary[name] = 'var(TPM) = 0.0'
			continue

		if pandas.api.types.is_numeric_dtype(column) :
			if sum_cutoff is not None and column.sum() < sum_cutoff :
				dictionary[name] = f'sum(TPM) < {sum_cutoff:.1f}'
				continue

			if max_cutoff is not None and column.max() < max_cutoff :
				dictionary[name] = f'max(TPM) < {max_cutoff:.1f}'
				continue

			if std_cutoff is not None and column.std() < std_cutoff :
				dictionary[name] = f'std(TPM) < {std_cutoff:.1f}'
				continue

			if avg_cutoff is not None and column.mean() < avg_cutoff :
				dictionary[name] = f'mean(TPM) < {avg_cutoff:.1f}'
				continue

			if px0_cutoff is not None :
				threshold = px0_cutoff[0]
				procentil = px0_cutoff[1]

				p = sum(item >= threshold for item in column) / len(column)
				s = f'{(100 * procentil):.0f}'

				if p < procentil :
					dictionary[name] = f'p{s}(TPM) < {threshold:.1f}'
					continue

	if len(dictionary) > 0 :
		item = list(dictionary.keys())[0]

		print(f'Problematic items found : {len(dictionary)} --> {item} ; {dictionary[item]}')
	else :
		print(f'Problematic items found : 0')

	return data, dictionary

def filter_genes (data : DataFrame, cutoff : Dict[str, Any] = None) -> Tuple[DataFrame, Dict[str, str]] :
	"""
	Doc
	"""

	data = data.copy()

	columns = data['Transcript'].tolist()

	data = data.iloc[:, 1:].transpose()
	data.columns = columns
	data = data.reset_index(names = 'Sample')

	return filter_samples(data = data, cutoff = cutoff)

def filter_genes_per_group (metadata : DataFrame, tpm : DataFrame, group : str = 'Tissue', cutoff : Dict[str, Any] = None) -> Tuple[List[str], Dict[str, str]] :
	"""
	Doc
	"""

	dictionary = dict()
	common = set()

	for index, (group, dataframe) in enumerate(metadata.groupby(group)) :
		samples = dataframe['Sample'].to_list()
		samples.insert(0, 'Transcript')

		_, items = filter_genes(
			data   = tpm[samples],
			cutoff = cutoff
		)

		dictionary.update({
			k : v + f' @ {group}'
			for k, v in items.items()
		})

		items = set(items.keys())

		if index == 0 :
			common = items
		else :
			common = common.intersection(items)

	common = list(common)

	if len(common) > 0 :
		print()
		print(f'Problematic items found : {len(common)} --> {common[0]} ; {dictionary[common[0]]}')
	else :
		print()
		print(f'Problematic items found : 0')

	return common, dictionary

def gene_lineplot (metadata : DataFrame, tpm : DataFrame, groupby : List[str], method : str, filename : str = None) -> None :
	"""
	Doc
	"""

	method = method.lower()

	sindex = 0
	eindex = 50

	genes = tpm.columns[1:eindex + 1].tolist()

	for group in groupby :
		fig, axis = matplotlib.pyplot.subplots(figsize = (16, 10))

		for label, dataframe in metadata.groupby(group) :
			if len(dataframe) == 0 :
				continue

			dataframe = tpm[dataframe['Sample']]
			matrix = dataframe.iloc[sindex :eindex, 1 :].to_numpy()

			dataframe = DataFrame.from_dict({
				'gene'  : genes,
				'mean'  : numpy.mean(matrix, axis = 1),
				'stdev' : numpy.std(matrix, axis = 1),
				'max'   : numpy.max(matrix, axis = 1),
				'p90'   : numpy.percentile(matrix, 90, axis = 1)
			})

			seaborn.lineplot(
				data      = dataframe,
				x         = 'gene',
				y         = method,
				linewidth = 1.5,
				alpha     = 0.65,
				ax        = axis,
				label     = label
			)

		axis.set_xticks(genes)
		axis.set_xticklabels(genes, rotation = 90)
		axis.set_xlabel('')
		axis.set_ylabel('TPM')

		if filename is not None :
			matplotlib.pyplot.savefig(
				filename + '.png',
				dpi    = 120,
				format = 'png'
			)
