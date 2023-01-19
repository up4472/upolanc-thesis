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

def log1p (x : numpy.ndarray, base : Union[int, str] = 2) -> numpy.ndarray :
	"""
	Doc
	"""

	if isinstance(base, int) :
		base = str(base)

	match base.lower() :
		case '2'  : return numpy.log2(x + 1)
		case 'e'  : return numpy.log1p(x)
		case '10' : return numpy.log10(x + 1)
		case _    : raise ValueError()

def expm1 (x : numpy.ndarray, base : Union[int, str] = 2) -> numpy.ndarray :
	"""
	Doc
	"""

	if isinstance(base, int) :
		base = str(base)

	match base.lower() :
		case '2'  : return numpy.exp2(x) - 1
		case 'e'  : return numpy.expm1(x)
		case '10' : return numpy.power(10, x) - 1
		case _    : raise ValueError()

def standardize (x : numpy.ndarray, axis : int = None) -> numpy.ndarray :
	"""
	Doc
	"""

	x = x - numpy.mean(x, axis = axis)
	x = x / numpy.std(x, axis = axis)

	return x

def normalize (x : numpy.ndarray, axis : int = None) -> numpy.ndarray :
	"""
	Doc
	"""

	x = x - numpy.min(x, axis = axis)
	x = x / numpy.max(x, axis = axis)

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
