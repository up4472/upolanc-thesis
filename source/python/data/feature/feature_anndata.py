from anndata import AnnData
from pandas  import DataFrame
from types   import FunctionType
from typing  import Dict
from typing  import List
from typing  import Tuple
from typing  import Union

import matplotlib
import numpy
import seaborn

from source.python.data.feature.feature_processing import boxcox1p
from source.python.data.feature.feature_processing import log1p
from source.python.data.feature.feature_processing import normalize
from source.python.data.feature.feature_processing import pca
from source.python.data.feature.feature_processing import standardize

def create_anndata (mat : DataFrame, obs : DataFrame) -> AnnData :
	"""
	Doc
	"""

	mat = mat.copy()
	obs = obs.copy()

	columns = mat['Transcript'].tolist()

	mat = mat.iloc[:, 1:].transpose()
	mat.columns = columns

	obs = obs.set_index('Sample', drop = True)
	obs.index.name = None

	data = AnnData(
		X   = mat.sort_index(ascending = True),
		obs = obs.sort_index(ascending = True)
	)

	data.X = data.X.astype(numpy.float64)

	return data

def show_structure (data : AnnData) -> None :
	"""
	Doc
	"""

	print(data)

def show_matrix (data : AnnData, layer : str = None, rows : int = 5, cols : int = 10) -> None :
	"""
	Doc
	"""

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	maxval = numpy.max(matrix)
	minval = numpy.min(matrix)
	stdval = numpy.std(matrix)
	medval = numpy.median(matrix)
	avgval = numpy.mean(matrix)

	defstring = '{:9,.1f}'
	stdstring = '{:.5f}'

	if layer is not None :
		defstring = '{: 9.5f}'

	print(f'   Max value : ' + defstring.format(maxval))
	print(f'  Mean value : ' + defstring.format(avgval) + ' \u00B1 ' + stdstring.format(stdval))
	print(f'Median value : ' + defstring.format(medval))
	print(f'   Min value : ' + defstring.format(minval))
	print()

	elements = numpy.size(matrix)

	posvalues = [0.0, 1.0, 10.0, 50.0, 100.0, 250.0]
	negvalues = [-x for x in reversed(posvalues)]
	number = '{: 6.1f}'

	if layer in ['log1p'] :
		posvalues = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]
		negvalues = [-x for x in reversed(posvalues)]
		number = '{: 3.1f}'
	if layer in ['boxcox1p'] :
		posvalues = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
		negvalues = [-x for x in reversed(posvalues)]
		number = '{: 3.1f}'
	if layer in ['standard'] :
		posvalues = [0.0, 0.3, 0.5, 1.0, 2.0, 5.0]
		negvalues = [-x for x in reversed(posvalues)]
		number = '{: 3.1f}'
	if layer in ['normal'] :
		posvalues = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
		negvalues = [-x for x in reversed(posvalues)]
		number = '{: 3.1f}'

	if layer in ['log1p', 'boxcox1p', 'standard', 'normal'] :
		print(f'Total elements        : {elements:11,d}')
	else :
		print(f'Total elements          : {elements:11,d}')

	for value in negvalues :
		if value <= minval :
			continue

		summary = (matrix < value).sum()
		percent = 100.0 * summary / elements

		print('Total elements < ' + number.format(value) + f' : {summary:11,d} [{percent:6.2f} %]')

	summary = (matrix == 0.0).sum()
	percent = 100.0 * summary / elements

	print(f'Total elements = ' + number.format(0.0) + f' : {summary:11,d} [{percent:6.2f} %]')

	for value in posvalues :
		if value >= maxval :
			continue

		summary = (matrix > value).sum()
		percent = 100.0 * summary / elements

		print('Total elements > ' + number.format(value) + f' : {summary:11,d} [{percent:6.2f} %]')

	print()
	print(matrix[:rows, :cols])

def compute_log1p (data : AnnData, store_into : str, layer : str = None, base : Union[int, str] = 2) -> AnnData :
	"""
	Doc
	"""

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	matrix = matrix.copy()
	matrix = log1p(x = matrix, base = base)

	data.layers[store_into] = matrix

	return data

def compute_boxcox1p (data : AnnData, store_into : str, eps : float = 1e-7, layer : str = None) -> Tuple[AnnData, Dict] :
	"""
	Doc
	"""

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	matrix = matrix.copy()
	matrix, lmbda = boxcox1p(x = matrix, eps = eps)

	data.layers[store_into] = matrix

	factors = {
		'lambda' : lmbda,
		'eps'    : eps,
	}

	return data, factors

def compute_standardized (data : AnnData, store_into : str, layer : str = None, axis : int = None) -> Tuple[AnnData, Dict] :
	"""
	Doc
	"""

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	matrix = matrix.copy()
	matrix, avg_value, std_value = standardize(x = matrix, axis = axis)

	data.layers[store_into] = matrix

	factors = {
		'std'        : std_value.tolist() if isinstance(std_value, numpy.ndarray) else std_value,
		'mean'       : avg_value.tolist() if isinstance(avg_value, numpy.ndarray) else avg_value,
		'transcript' : data.var.index.to_list()
	}

	return data, factors

def compute_normalized (data : AnnData, store_into : str, layer : str = None) -> Tuple[AnnData, Dict] :
	"""
	Doc
	"""

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	matrix = matrix.copy()
	matrix, min_value, max_value = normalize(x = matrix)

	data.layers[store_into] = matrix

	factors = {
		'min' : min_value,
		'max' : max_value,
	}

	return data, factors

def compute_pca (data : AnnData, store_into : str, layer : str = None, components : int = 50) -> AnnData :
	"""
	Doc
	"""

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	matrix = matrix.copy()
	matrix = pca(matrix = matrix, components = components)

	data.obsm[store_into] = matrix

	return data

def tpm_histplot (data : AnnData, layer : str, function : FunctionType, filters : List[Tuple[FunctionType, float]] = None, filename : str = None) -> None :
	"""
	Doc
	"""

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	if filters is not None :
		mask = False

		for func, threshold in filters :
			vals = func(matrix, axis = 0)

			fail = vals < threshold
			mask = mask | fail

			print(f'Removed [{sum(fail):6,d}] genes due to [{func.__module__}.{func.__name__}] < [{threshold:.2f}]')

		print()

		matrix = matrix[:, ~mask]

	vector = function(matrix, axis = 0)

	data = DataFrame.from_dict({
		'Values' : vector
	})

	_, axis = matplotlib.pyplot.subplots(figsize = (16, 10))

	p10 = numpy.percentile(vector, 10)
	p30 = numpy.percentile(vector, 30)
	p70 = numpy.percentile(vector, 70)
	p90 = numpy.percentile(vector, 90)

	n10 = len(vector[(vector < p10)])
	n30 = len(vector[(vector < p30)])
	n70 = len(vector[(vector > p70)])
	n90 = len(vector[(vector > p90)])

	if layer is None :
		print(f'Function : {function.__module__}.{function.__name__}')
		print(f'Genes below 10th percentile [{p10: 11,.2f}] : {n10:6,d}')
		print(f'Genes below 30th percentile [{p30: 11,.2f}] : {n30:6,d}')
		print(f'Genes above 70th percentile [{p70: 11,.2f}] : {n70:6,d}')
		print(f'Genes above 90th percentile [{p90: 11,.2f}] : {n90:6,d}')
		print()
	else :
		print(f'Function : {function.__module__}.{function.__name__}')
		print(f'Genes below 10th percentile [{p10: 11.6f}] : {n10:6,d}')
		print(f'Genes below 30th percentile [{p30: 11.6f}] : {n30:6,d}')
		print(f'Genes above 70th percentile [{p70: 11.6f}] : {n70:6,d}')
		print(f'Genes above 90th percentile [{p90: 11.6f}] : {n90:6,d}')
		print()

	seaborn.histplot(
		data  = data,
		x     = 'Values',
		alpha = 0.9,
		color = '#799FCB',
		ax    = axis,
		kde   = False
	)

	axis.set_xlabel(f'{function.__name__}')

	if filename is not None :
		matplotlib.pyplot.savefig(
			f'{filename}.png',
			dpi    = 120,
			format = 'png'
		)

def gene_boxplot (data : AnnData, groupby : str, layer : str = None, transcript : Union[int, str] = 0, filename : str = None) -> None :
	"""
	Doc
	"""

	genes = data.var.index.to_list()

	if isinstance(transcript, int) :
		columns    = transcript
		transcript = data.var.index.to_list()[transcript]
	else :
		columns = genes.index(transcript)

	if layer is None :
		mat = data.X[:, columns]
	else :
		mat = data.layers[layer][:, columns]

	dataframe = DataFrame(mat, columns = [transcript])
	dataframe.insert(0, groupby, data.obs[groupby].to_list())

	_, axis = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.boxplot(data = dataframe, x = transcript, y = groupby, ax = axis)

	axis.set_title(transcript)
	axis.set_ylabel('')
	axis.set_xlabel(layer)

	if layer is None :
		axis.set_xlabel('TPM')
	if layer == 'log1p' :
		axis.set_xlabel('log(TPM + 1)')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			dpi    = 120,
			format = 'png'
		)
