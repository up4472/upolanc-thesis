from typing import Tuple

import scipy
import numpy

def interquartile_range (data : numpy.ndarray, k : float = 1.5, axis : int = None) -> Tuple[numpy.ndarray, float, float, float] :
	"""
	Doc
	"""

	q1 = numpy.percentile(data, 25, axis = axis, method = 'midpoint')
	q3 = numpy.percentile(data, 75, axis = axis, method = 'midpoint')

	iqr = q3 - q1

	upper = q3 + k * iqr
	lower = q1 - k * iqr
	total = numpy.size(data, axis = axis)

	if numpy.ndim(data) == 2 :
		threshold = data.copy()

		for i in range(numpy.size(data, axis = 0)) :
			threshold_upper = data[i, :] < upper[i]
			threshold_lower = data[i, :] > lower[i]

			threshold[i, :] = numpy.logical_and(threshold_upper, threshold_lower)

	elif numpy.ndim(data) == 1 :
		threshold_upper = data < upper
		threshold_lower = data > lower

		threshold = numpy.logical_and(threshold_upper, threshold_lower)

	else :
		raise ValueError()

	data = numpy.where(threshold, data, numpy.nan)
	keep = numpy.sum(threshold, axis = axis)

	return data, lower, upper, keep / total

def zscore (data : numpy.ndarray, z : float = 3, axis : int = None, ddof : int = 0) -> Tuple[numpy.ndarray, float, float, float] :
	"""
	Doc
	"""

	score = scipy.stats.zscore(data, ddof = ddof, axis = axis)

	mean = numpy.mean(data, axis = axis)
	std = numpy.std(data, axis = axis)

	upper = mean + z * std
	lower = mean - z * std
	total = numpy.size(data, axis = axis)

	threshold = score < z

	data = numpy.where(threshold, data, numpy.nan)
	keep = numpy.sum(threshold, axis = axis)

	return data, lower, upper, keep / total
