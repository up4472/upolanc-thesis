from pandas import DataFrame
from typing import List
from typing import Union

import scipy
import seaborn
import matplotlib
import numpy

from source.python.data.stats.stats_normality  import anderson
from source.python.data.stats.stats_normality  import normaltest
from source.python.data.stats.stats_normality  import shapiro
from source.python.data.stats.stats_statistics import interquartile_range
from source.python.data.stats.stats_statistics import zscore

def generate_basic_statistics (data : Union[DataFrame, numpy.ndarray], transcript : List[str], tissue : str, axis : int = 1) -> DataFrame :
	"""
	Doc
	"""

	if isinstance(data, DataFrame) :
		matrix = data.iloc[:, 1:].values.copy()
	else :
		matrix = data.copy()

	if tissue is None : tissue = 'Global'
	else              : tissue = tissue.title()

	return DataFrame.from_dict({
		'Transcript' : transcript,
		'Tissue'     : tissue,
		'Count'      : numpy.size(matrix, axis = axis),
		'Mean'       : numpy.mean(matrix, axis = axis),
		'Median'     : numpy.median(matrix, axis = axis),
		'StDev'      : numpy.std(matrix, axis = axis),
		'Var'        : numpy.var(matrix, axis = axis),
		'Min'        : numpy.min(matrix, axis = axis),
		'Max'        : numpy.max(matrix, axis = axis),
		'Range'      : numpy.ptp(matrix, axis = axis),
		'MAD'        : scipy.stats.median_abs_deviation(matrix, axis = axis, nan_policy = 'omit')
	})

def genearte_advance_statistics (data : Union[DataFrame, numpy.ndarray], transcript : List[str], tissue : str, axis : int = 1) -> DataFrame :
	"""
	Doc
	"""

	if isinstance(data, DataFrame) :
		matrix = data.iloc[:, 1:].values.copy()
	else :
		matrix = data.copy()

	if tissue is None : tissue = 'Global'
	else              : tissue = tissue.title()

	res_zscore = zscore(data = matrix, z = 3, ddof = 0, axis = axis)
	res_iqr    = interquartile_range(data = matrix, k = 1.5, axis = axis)

	return DataFrame.from_dict({
		'Transcript'     : transcript,
		'Tissue'         : tissue,
		'ZScore-Lower'   : res_zscore[1],
		'ZScore-Upper'   : res_zscore[2],
		'ZScore-Percent' : res_zscore[3],
		'IQR-Lower'      : res_iqr[1],
		'IQR-Upper'      : res_iqr[2],
		'IQR-Percent'    : res_iqr[3]
	})

def generate_normality_statistics (data : Union[DataFrame, numpy.ndarray], transcript : List[str], tissue : str) -> DataFrame :
	"""
	Doc
	"""

	if isinstance(data, DataFrame) :
		matrix = data.iloc[:, 1:].values.copy()
	else :
		matrix = data.copy()

	if tissue is None : tissue = 'Global'
	else              : tissue = tissue.title()

	data = {
		'Transcript'             : transcript,
		'Tissue'                 : tissue,
		'Shapiro-Statistic'      : list(),
		'Shapiro-pValue'         : list(),
		'NormalTest-Statistic'   : list(),
		'NormalTest-pValue'      : list(),
		'Anderson-Statistic'     : list(),
		'Anderson-CriticalValue' : list(),
		'Anderson-Significance'  : list(),
	}

	for index, transcript in enumerate(transcript) :
		array = matrix[index, :]

		result = shapiro(data = array)
		data['Shapiro-Statistic'].append(result[0])
		data['Shapiro-pValue'   ].append(result[1])

		result = normaltest(data = array)
		data['NormalTest-Statistic'].append(result[0])
		data['NormalTest-pValue'   ].append(result[1])

		result = anderson(data = array, dist = 'norm')
		data['Anderson-Statistic'    ].append(result[0])
		data['Anderson-CriticalValue'].append(result[1])
		data['Anderson-Significance' ].append(result[2])

	return DataFrame.from_dict(data)

def statistic_histplot (data : DataFrame, x : str, label : str = None, title : str = None, filename : str = None) -> None :
	"""
	Doc
	"""

	fig, axis = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.histplot(
		data = data,
		x    = x,
		kde  = True,
		ax   = axis
	)

	if title is not None : axis.set_title(title)
	if label is not None : axis.set_xlabel(label)

	axis.set_ylabel('Count')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			dpi    = 120,
			format = 'png'
		)
