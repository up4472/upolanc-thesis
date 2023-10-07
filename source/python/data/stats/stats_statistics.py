from typing import Tuple

import scipy
import numpy

def sanity_check_filter (data : numpy.ndarray, threshold : numpy.ndarray, axis : int = None, n : int = None) -> Tuple[numpy.ndarray, numpy.ndarray] :
	"""
	Doc
	"""

	total = numpy.size(data, axis = axis)

	if n is None or n <= 0 :
		keep = numpy.sum(threshold, axis = axis)
		data = numpy.where(threshold, data, numpy.nan)

		return data, keep / total

	keep = numpy.sum(threshold, axis = axis)
	keep = keep / total
	flag = keep > (n / total)

	if flag.all() :
		data = numpy.where(threshold, data, numpy.nan)

		return data, keep

	index = numpy.where(flag == True)

	data[:, index] = numpy.where(threshold[:, index], data[:, index], numpy.nan)

	nonnan  = numpy.count_nonzero(~numpy.isnan(data), axis = axis)
	percent = nonnan / total

	return data, percent

def interquartile_range (data : numpy.ndarray, k : float = 1.5, axis : int = None, n : int = None) -> Tuple[numpy.ndarray, float, float, numpy.ndarray] :
	"""
	Doc
	"""

	q1 = numpy.percentile(data, 25, axis = axis, method = 'midpoint')
	q3 = numpy.percentile(data, 75, axis = axis, method = 'midpoint')

	iqr = q3 - q1

	upper = q3 + k * iqr
	lower = q1 - k * iqr

	if numpy.ndim(data) == 2 :
		threshold = numpy.zeros_like(data, dtype = numpy.bool)

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

	data, keep = sanity_check_filter(
		data      = data,
		threshold = threshold,
		axis      = axis,
		n         = n
	)

	return data, lower, upper, keep

def zscore (data : numpy.ndarray, z : float = 3, axis : int = None, ddof : int = 0, n : int = None) -> Tuple[numpy.ndarray, float, float, numpy.ndarray] :
	"""
	Doc
	"""

	score = scipy.stats.zscore(data, ddof = ddof, axis = axis)

	mean = numpy.mean(data, axis = axis)
	std  = numpy.std(data,  axis = axis)

	upper = mean + z * std
	lower = mean - z * std

	upper_threshold = score <  z
	lower_threshold = score > -z

	threshold = numpy.logical_and(upper_threshold, lower_threshold)

	data, keep = sanity_check_filter(
		data      = data,
		threshold = threshold,
		axis      = axis,
		n         = n
	)

	return data, lower, upper, keep
