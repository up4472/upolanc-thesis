from typing import Tuple

import scipy
import numpy

def normaltest (data : numpy.ndarray) -> Tuple[float, float, float] :
	"""
	Doc
	"""

	statistic = numpy.nan
	pvalue    = numpy.nan

	if len(data) > 3 :
		result = scipy.stats.normaltest(data, axis = None, nan_policy = 'omit')

		statistic = result.statistic # noqa
		pvalue    = result.pvalue    # noqa

	return statistic, pvalue, numpy.nan

def shapiro (data : numpy.ndarray) -> Tuple[float, float, float] :
	"""
	Doc
	"""

	statistic = numpy.nan
	pvalue = numpy.nan

	if len(data) > 3 :
		result = scipy.stats.shapiro(data)

		statistic = result.statistic # noqa
		pvalue    = result.pvalue    # noqa

	return statistic, pvalue, numpy.nan

def anderson (data : numpy.ndarray, dist : str = 'norm') -> Tuple[float, float, float] :
	"""
	Doc
	"""

	statistic    = numpy.nan
	critical     = numpy.nan
	significance = numpy.nan

	if len(data) > 3 :
		result = scipy.stats.anderson(data, dist = dist)

		statistic    = result.statistic
		critical     = numpy.nan
		significance = numpy.nan

		for cv, sl in zip(result.critical_values, result.significance_level) :
			critical     = cv
			significance = sl

			if critical >= statistic :
				break

	return statistic, critical, significance
