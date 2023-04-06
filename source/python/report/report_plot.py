from pandas import DataFrame
from typing import Any
from typing import Dict

import seaborn
import matplotlib

def models_bert_r2 (data : Dict[str, Any], mode : str = 'regression', step : str = 'iteration', x : int = None, y : int = None, filename : str = None) :
	"""
	Doc
	"""

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	per_step  = ('step', 'Step')
	per_epoch = ('epoch', 'Epoch')

	if   step == 'iteration' : xcolumn = per_step
	elif step == 'step'      : xcolumn = per_step
	elif step == 'batch'     : xcolumn = per_step
	elif step == 'epoch'     : xcolumn = per_epoch
	else                     : xcolumn = per_step

	for name, dataframe in data[mode].items() :
		name = name[17:]

		if y is None : y = len(dataframe)
		if x is None : x = 0

		dataframe = dataframe.head(n = y)
		dataframe = dataframe.tail(n = y - x)

		dataframe['epoch'] = dataframe['step'] / 480

		if   step == 'iteration' : xcolumn = ('step',  'Step')
		elif step == 'step'      : xcolumn = ('step',  'Step')
		elif step == 'epoch'     : xcolumn = ('epoch', 'Epoch')
		else                     : xcolumn = ('step',  'Step')

		seaborn.lineplot(
			data  = dataframe,
			x     = xcolumn[0],
			y     = 'eval_r2',
			ax    = ax,
			alpha = 0.8,
			label = name
		)

	ax.set_ylabel('Eval R2')
	ax.set_xlabel(xcolumn[1])

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format = 'png',
			dpi    = 120
		)

def model_bert_r2 (dataframe : DataFrame, x : int = None, y : int = None, filename : str = None) -> None :
	"""
	Doc
	"""

	if y is None : y = len(dataframe)
	if x is None : x = 0

	dataframe = dataframe.head(n = y)
	dataframe = dataframe.tail(n = y - x)

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.lineplot(
		data = dataframe,
		x    = 'step',
		y    = 'eval_r2',
		ax   = ax
	)

	ax.set_ylabel('Eval R2')
	ax.set_xlabel('Step')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format = 'png',
			dpi    = 120
		)

def model_bert_r2_keyword (data : Dict[str, Any], mode : str = 'regression', model : str = 'bertfc3-def', kmer : int = 6, sequence : str = 'promoter', epochs : int = 500, target : str = 'global-mean', x : int = 0, y : int = 1000) -> None :
	"""
	Doc
	"""

	key = 'model-{}-{}-{}-{}-{:04d}-{}'.format(mode, model, kmer, sequence, epochs, target)

	model_bert_r2(
		dataframe = data[mode][key],
		x         = x,
		y         = y,
		filename  = None
	)
