from pandas import DataFrame
from typing import Dict
from typing import List

import scipy
import seaborn
import matplotlib
import numpy

def create_data (data : Dict[str, Dict]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	if data is not None :
		return data

	return {
		'Count'       : dict(),
		'Min'         : dict(),
		'Max'         : dict(),
		'Mean'        : dict(),
		'Median'      : dict(),
		'StDev'       : dict(),
		'Variance'    : dict(),
		'Range'       : dict(),
		'MAD'         : dict(),
		'Shapiro'     : dict(),
		'Shapiro.p'   : dict(),
		'Anderson'    : dict(),
		'Anderson.p'  : dict(),
		'DAgostino'   : dict(),
		'DAgostino.p' : dict()
	}

def get_statistics_for (dataframe : DataFrame, transcripts : List[str], group : str, data : Dict[str, Dict] = None, axis : int = 1) -> Dict[str, Dict] :
	"""
	Doc
	"""

	matrix = dataframe.iloc[:, 1:].values

	if group is None : group = 'Global'
	else             : group = group.title()

	data = create_data(data = data)

	mat_min      = numpy.min(matrix, axis = axis)
	mat_max      = numpy.max(matrix, axis = axis)
	mat_mean     = numpy.mean(matrix, axis = axis)
	mat_median   = numpy.median(matrix, axis = axis)
	mat_stdev    = numpy.std(matrix, axis = axis)
	mat_variance = numpy.var(matrix, axis = axis)
	mat_range    = numpy.ptp(matrix, axis = axis)
	mat_mad      = scipy.stats.median_abs_deviation(matrix, axis = axis)

	for index, transcript in enumerate(transcripts) :
		array = matrix[index, :]
		key   = (transcript, group)

		data['Count'   ][key] = len(array)
		data['Min'     ][key] = mat_min[index]
		data['Max'     ][key] = mat_max[index]
		data['Mean'    ][key] = mat_mean[index]
		data['Median'  ][key] = mat_median[index]
		data['StDev'   ][key] = mat_stdev[index]
		data['Variance'][key] = mat_variance[index]
		data['Range'   ][key] = mat_range[index]
		data['MAD'     ][key] = mat_mad[index]

		if len(array) < 4 :
			data['Shapiro'][key]   = numpy.nan
			data['Anderson'][key]  = numpy.nan
			data['DAgostino'][key] = numpy.nan

			data['Shapiro.p'][key]   = numpy.nan
			data['Anderson.p'][key]  = numpy.nan
			data['DAgostino.p'][key] = numpy.nan

		else :
			shapiro   = scipy.stats.shapiro(array)
			anderson  = scipy.stats.anderson(array, dist = 'norm')
			dagostino = scipy.stats.normaltest(array)

			curr    = anderson.statistic
			last_sl = None

			for cv, sl in zip(anderson.critical_values, anderson.significance_level) :
				last_cv = cv
				last_sl = sl

				if last_cv >= curr :
					break

			data['Shapiro'][key]   = shapiro.statistic   # noqa
			data['Anderson'][key]  = anderson.statistic  # noqa
			data['DAgostino'][key] = dagostino.statistic # noqa

			data['Shapiro.p'][key]   = shapiro.pvalue   # noqa
			data['Anderson.p'][key]  = last_sl          # noqa
			data['DAgostino.p'][key] = dagostino.pvalue # noqa

	return data

def get_statistics_dataframe (data : Dict[str, Dict]) -> DataFrame :
	"""
	Doc
	"""

	dataframe = DataFrame.from_dict(data)
	dataframe.index = dataframe.index.set_names(['Transcript', 'Tissue'])

	dataframe = dataframe.reset_index()

	dataframe.set_index(['Transcript', 'Tissue'], inplace = True)
	dataframe.sort_index(inplace = True)

	return dataframe

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
