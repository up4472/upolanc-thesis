from anndata  import AnnData
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
from typing   import List

import distinctipy
import itertools
import matplotlib
import numpy

def select_genes (matrix : numpy.ndarray, threshold : float = 0.0, n : int = 1000, decay : float = 1.0, xoffset : float = 5.0, yoffset : float = 0.02) : # noqa
	"""
	Doc
	"""

	raise NotImplementedError()

def compute_tsne (data : AnnData, features : str, store_into : str = 'tsne', perplexities : List[int] = None) -> AnnData :
	"""
	Doc
	"""

	if perplexities is None :
		perplexities = [50, 250]

	affinities = affinity.Multiscale(
		data = data.obsm[features],
		perplexities = perplexities,
		metric = 'cosine',
		n_jobs = 8,
		random_state = 3
	)

	embedding = initialization.pca(
		X = data.obsm[features],
		random_state = 42
	)

	embedding = TSNEEmbedding(
		embedding = embedding,
		affinities = affinities,
		negative_gradient_method = 'fft',
		n_jobs = 8
	)

	embedding = embedding.optimize(n_iter = 250, exaggeration = 12, momentum = 0.5, inplace = True)
	embedding = embedding.optimize(n_iter = 750, exaggeration = 1,  momentum = 0.8, inplace = True)

	data.obsm[store_into] = embedding

	return data

def visualize (data : AnnData, feature : str, groupby : str, filename : str = None, alpha : float = 0.5, size : int = 1) -> None :
	"""
	Doc
	"""

	figure, axis = matplotlib.pyplot.subplots(figsize = (16, 10), dpi = 120)

	matplotlib.pyplot.subplots_adjust(right = 0.8)

	nx = data.obs[groupby].nunique()

	colors = distinctipy.get_colors(n_colors = nx, rng = 3)
	colors = [distinctipy.get_hex(color = item) for item in colors]

	colors = dict(itertools.zip_longest(
		data.obs[groupby].value_counts().sort_values(ascending = False).index,
		colors,
		fillvalue = colors[-1]
	))

	for item in data.obs[groupby].unique() :
		if item not in colors :
			print(f'Error : failed to process color for {item}')

	x = data.obsm[feature]
	y = data.obs[groupby]

	ordered = list(colors.keys())
	classes = [item for item in ordered if item in numpy.unique(y)]

	axis.scatter(
		x          = x[:, 0],
		y          = x[:, 1],
		c          = list(map(colors.get, y)),
		rasterized = True,
		alpha      = alpha,
		s          = size
	)

	axis.set_xticks([])
	axis.set_yticks([])
	axis.axis('off')

	legend_handles = [
		matplotlib.lines.Line2D(
			xdata           = [],
			ydata           = [],
			marker          = 's',
			color           = 'w',
			markerfacecolor = colors[i],
			ms              = 10,
			alpha           = 1,
			linewidth       = 0,
			label           = i,
			markeredgecolor = 'k'
		)
		for i in classes
	]

	axis.legend(
		handles        = legend_handles,
		loc            = 'center left',
		bbox_to_anchor = (1.0, 0.5),
		frameon        = False
	)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			dpi    = 120,
			format = 'png'
		)
