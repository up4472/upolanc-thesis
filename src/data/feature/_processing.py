from typing import Tuple
from typing import Union

import numpy
import scipy

def boxcox1p (x : numpy.ndarray, eps : float = 1e-7) -> Tuple[numpy.ndarray, float] :
	"""
	Doc
	"""

	array = x.copy().flatten()
	array = array + eps

	_, lmbda = scipy.stats.boxcox(array, lmbda = None)
	x = scipy.special.boxcox1p(x, lmbda)

	return x, lmbda

def boxcox1p_inv (x : numpy.ndarray, lmbda : float) -> numpy.ndarray :
	"""
	Doc
	"""

	return scipy.special.inv_boxcox1p(x, lmbda)

def log1p (x : numpy.ndarray, base : Union[int, str] = 2) -> numpy.ndarray :
	"""
	Doc
	"""

	if isinstance(base, int) :
		base = str(base)

	base = base.lower()

	if   base == '2'  : return numpy.log2(x + 1)
	elif base == 'e'  : return numpy.log1p(x)
	elif base == '10' : return numpy.log10(x + 1)
	else : raise ValueError()

def log1p_inv (x : numpy.ndarray, base : Union[int, str] = 2) -> numpy.ndarray :
	"""
	Doc
	"""

	if isinstance(base, int) :
		base = str(base)

	base = base.lower()

	if   base == '2'  : return numpy.exp2(x) - 1
	elif base == 'e'  : return numpy.expm1(x)
	elif base == '10' : return numpy.power(10, x) - 1
	else : raise ValueError()

def normalize (x : numpy.ndarray) -> Tuple[numpy.ndarray, float, float] :
	"""
	Doc
	"""

	min_value = numpy.min(x, axis = None)
	x = x - min_value
	max_value = numpy.max(x, axis = None)
	x = x / max_value

	return x, min_value, max_value

def normalize_inv (x : numpy.ndarray, min_value : float, max_value : float) -> numpy.ndarray :
	"""
	Doc
	"""

	x = x * max_value
	x = x + min_value

	return x

def standardize (x : numpy.ndarray, axis : int = None) -> numpy.ndarray :
	"""
	Doc
	"""

	x = x - numpy.mean(x, axis = axis)
	x = x / numpy.std(x, axis = axis)

	return x

def pca (matrix : numpy.ndarray, components : int = 50) -> numpy.ndarray :
	"""
	Doc
	"""

	U, S, V = numpy.linalg.svd(
		a = matrix,
		full_matrices = False
	)

	V = numpy.sum(V, axis = 1)
	D = numpy.diag(S)
	A = numpy.argsort(S)[: :-1]

	U[:, V < 0] *= -1

	matrix = numpy.dot(U, D)
	matrix = matrix[:, A][:, :components]

	return matrix
