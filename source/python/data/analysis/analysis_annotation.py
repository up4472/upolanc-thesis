from pandas import DataFrame
from typing import List
from typing import Optional

import matplotlib
import numpy
import scipy
import seaborn

def show (data : DataFrame, query : str = None, query_by : str = None, rows : Optional[int] = 5, cols : int = 10) -> DataFrame :
	"""
	Doc
	"""

	if query is not None :
		data = data.loc[data[query_by].isin([query])]

	print(f'Number of rows : {data.shape[0]}')
	print(f'Number of cols : {data.shape[1]}')

	if rows is None :
		rows = len(data)

	if len(data.columns) > cols :
		return data.iloc[:, :cols].head(n = rows)

	return data.head(n = rows)

def inspect_columns (data : DataFrame, columns : List[str] = None, items : int = 5) -> DataFrame :
	"""
	Doc
	"""

	if columns is None :
		columns = data.columns.tolist()

	dtypes = list()
	ncount = list()
	ucount = list()

	print('Unique values per column :')
	print()

	for name in columns :
		column = data[name]
		unique = column.unique().tolist()
		null   = column.isnull().sum()

		dtypes.append(str(column.dtypes))
		ncount.append(null)
		ucount.append(len(unique))

		if len(unique) > items :
			print(f' - {name:10s} : [{len(unique):7,d}] ' + ' '.join(str(val) for val in unique[:items]) + ' ...')
		else :
			print(f' - {name:10s} : [{len(unique):7,d}] ' + ' '.join(str(val) for val in unique[:items]))

	return DataFrame(
		data    = [dtypes, ncount, ucount],
		columns = columns,
		index   = ['Datatype', 'Null', 'Unique']
	)

def type_distribution (data : DataFrame, groupby : str, regions : List[str] = None) -> DataFrame :
	"""
	Doc
	"""

	if regions is None :
		regions = data['Type'].unique().tolist()

	data = data[data['Type'].isin(regions)].copy()

	matrix = numpy.zeros(shape = (len(regions), 7), dtype = int)
	column = ['Count', *[str(value) for value in range(5)], f'5+']

	for _, dataframe in data.groupby(groupby) :
		lengths = [len(dataframe[dataframe['Type'] == region]) for region in regions]

		for index, length in enumerate(lengths) :
			matrix[index, 0] = matrix[index, 0] + length

			if length == 0 : matrix[index, 1] = matrix[index, 1] + 1
			if length == 1 : matrix[index, 2] = matrix[index, 2] + 1
			if length == 2 : matrix[index, 3] = matrix[index, 3] + 1
			if length == 3 : matrix[index, 4] = matrix[index, 4] + 1
			if length == 4 : matrix[index, 5] = matrix[index, 5] + 1
			if length >= 5 : matrix[index, 6] = matrix[index, 6] + 1

	return DataFrame(
		data    = matrix,
		columns = column,
		index   = regions
	)

def group_regions (data : DataFrame, groupby : str, regions : List[str] = None) -> DataFrame :
	"""
	Doc
	"""

	if regions is None :
		regions = data['Type'].unique().tolist()

	entries = list()

	for group, dataframe in data.groupby(groupby) :
		f0 = dataframe['Seq'].iloc[0]
		f1 = dataframe['Strand'].iloc[0]
		f3 = dataframe['Gene'].iloc[0]
		f4 = dataframe['Transcript'].iloc[0]

		for region in regions :
			f2 = region
			f5 = dataframe['Exon'].iloc[0]
			f6 = dataframe['Parent'].iloc[0]

			data = dataframe.loc[dataframe['Type'].isin([region])]

			f7 = -1 if len(data) == 0 else data['Start'].min()
			f8 = -1 if len(data) == 0 else data['End'].max()
			f9 =  0 if len(data) == 0 else data['Length'].sum()

			entries.append([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, len(data)])

	return DataFrame(
		data     = entries,
		columns = ['Seq', 'Strand', 'Type', 'Gene', 'Transcript', 'Exon', 'Parent', 'Start', 'End', 'Length', 'Regions']
	)

def length_statistics (data : DataFrame, regions : List[str] = None) -> DataFrame :
	"""
	Doc
	"""

	if regions is None :
		regions = data['Type'].unique().tolist()

	dataframes = [data.loc[data['Type'] == region] for region in regions]

	columns = list()
	indices = list()

	for dataframe, region in zip(dataframes, regions) :
		columns.append(dataframe['Length'].describe())
		indices.append(region)

	dataframe = DataFrame(data = columns, index = indices)
	dataframe.columns = [column.title() for column in dataframe.columns]

	return dataframe

def length_histplot (data : DataFrame, value : str, vline : int, linewidth : int = 1, show_xlabel : bool = True, show_ylabel : bool = True, filename : str = None) -> None :
	"""
	Doc
	"""

	dataframe = data.loc[data['Type'] == value]
	dataframe = dataframe[dataframe['Length'] > 0]

	figure, axis = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.histplot(
		x         = 'Length',
		data      = dataframe,
		ax        = axis,
		color     = '#799FCB',
		log_scale = True,
		alpha     = 0.9,
		kde       = False
	)

	if vline > 0 :
		percentile = scipy.stats.percentileofscore(dataframe['Length'], vline, nan_policy = 'omit')

		x = 1.05 * vline
		y = 0.95 * axis.get_ylim()[-1]

		axis.axvline(vline, color = '#F0665E', linewidth = linewidth, alpha = 0.9)
		axis.text(x, y, f'{percentile:.1f}%', color = '#F0665E', alpha = 0.9)

	if show_xlabel : axis.set_xlabel(value)
	else           : axis.set_xlabel(None)

	if show_ylabel : ...
	else           : axis.set_ylabel(None)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			dpi         = 120,
			format      = 'png',
			bbox_inches = 'tight',
			pad_inches  = 0
		)
