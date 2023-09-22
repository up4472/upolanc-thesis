from pandas import DataFrame
from typing import List

import matplotlib
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
			print(f' - {name:12s} : [{len(unique):5,d}] ' + ' '.join(str(val) for val in unique[:items]) + ' ...')
		else :
			print(f' - {name:12s} : [{len(unique):5,d}] ' + ' '.join(str(val) for val in unique[:items]))

	return DataFrame(
		data    = [dtypes, ncount, ucount],
		columns = columns,
		index   = ['Datatype', 'Null', 'Unique']
	)

def distribution_barplot (data : DataFrame, group : str, filename : str = None) -> None :
	"""
	Doc
	"""

	unique = data[group].unique()
	counts = [len(data.loc[data[group] == name]) for name in unique]

	df = DataFrame.from_dict({
		'Class' : unique,
		'Count' : counts
	}).sort_values('Count', ascending = False)

	fig, axis = matplotlib.pyplot.subplots(figsize = (16, 10))
	fig.tight_layout()

	graph = seaborn.barplot(
		data   = df,
		x      = 'Class',
		y      = 'Count',
		hue    = 'Class',
		width  = 0.8,
		ax     = axis,
		alpha  = 0.9,
		dodge  = False
	)

	graph.set(xticklabels = [])
	graph.set(xlabel = None)
	graph.tick_params(bottom = False)

	axis.set_xlabel(None)
	axis.set_ylabel(None)
	axis.legend(loc = 'upper right')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			dpi         = 120,
			format      = 'png',
			bbox_inches = 'tight',
			pad_inches  = 0
		)
