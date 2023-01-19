from typing import Dict
from typing import List

import matplotlib
import numpy
import seaborn

def lineplot (values : List[numpy.ndarray | List], labels : List[str], xlabel : str, ylabel : str, title : str = None, filename : str = None) -> None :
	"""
	Doc
	"""

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	x = numpy.arange(1, 1 + len(values[0]))

	for y, label in zip(values, labels) :
		seaborn.lineplot(x = x, y = y, label = label, ax = ax)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	if title is not None :
		ax.set_title(title)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			dpi    = 120,
			format = 'png'
		)

def show_metric_grid (report : Dict[str, Dict], mode : str = 'train', filename : str = None) -> None :
	"""
	Doc
	"""

	metrics = list(report[mode]['metric'].keys())
	names   = list()

	for metric in metrics :
		match metric.lower() :
			case 'mae'        : names.append('MAE')
			case 'mse'        : names.append('MSE')
			case 'huber'      : names.append('Huber')
			case 'smooth-mae' : names.append('Smooth MAE')
			case 'ce'         : names.append('Cross Entropy')
			case 'cce'        : names.append('Categorical Cross Entropy')
			case 'nll'        : names.append('Negative Log Likelihood')
			case 'kl'         : names.append('Kullback-Leibler Divergence')
			case 'r2'         : names.append('R2 Score')
			case 'acc'        : names.append('Accuracy')
			case _            : names.append('?')

	n = len(names)

	match n :
		case 1  : nrows, ncols, gridlike = 1, 1, False
		case 2  : nrows, ncols, gridlike = 1, 2, False
		case 3  : nrows, ncols, gridlike = 1, 3, False
		case 4  : nrows, ncols, gridlike = 2, 2, True
		case 5  : nrows, ncols, gridlike = 2, 3, True
		case 6  : nrows, ncols, gridlike = 2, 3, True
		case 7  : nrows, ncols, gridlike = 2, 4, True
		case 8  : nrows, ncols, gridlike = 2, 4, True
		case 9  : nrows, ncols, gridlike = 3, 3, True
		case 10 : nrows, ncols, gridlike = 3, 4, True
		case 11 : nrows, ncols, gridlike = 3, 4, True
		case 12 : nrows, ncols, gridlike = 3, 4, True
		case 13 : nrows, ncols, gridlike = 4, 4, True
		case 14 : nrows, ncols, gridlike = 4, 4, True
		case 15 : nrows, ncols, gridlike = 4, 4, True
		case 16 : nrows, ncols, gridlike = 4, 4, True
		case 17 : nrows, ncols, gridlike = 5, 4, True
		case 18 : nrows, ncols, gridlike = 5, 4, True
		case 19 : nrows, ncols, gridlike = 5, 4, True
		case 20 : nrows, ncols, gridlike = 5, 4, True
		case _  : nrows, ncols, gridlike = n, 1, False

	_, ax = matplotlib.pyplot.subplots(nrows, ncols, figsize = (16 * ncols, 10 * nrows))

	for index, (metric, name) in enumerate(zip(metrics, names)):
		metric = report[mode]['metric'][metric]
		metric = numpy.array([x.mean() for x in metric])

		if gridlike :
			axis = ax[index // ncols, index % ncols]
		else :
			axis = ax[index]

		seaborn.lineplot(x = numpy.arange(1, 1 + len(metric)), y = metric, ax = axis)

		axis.set_xlabel('Epoch')
		axis.set_ylabel('Loss')
		axis.set_title(name)

	for index in range(n, nrows * ncols) :
		axis = ax[index // ncols, index % ncols] if gridlike else ax[index]
		axis.axis('off')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-metrics.png',
			dpi    = 120,
			format = 'png'
		)

def show_metric (report : Dict[str, Dict], mode : str, metric : str, title : str = None, filename : str = None) -> None :
	"""
	Doc
	"""

	values = report[mode]['metric'][metric]
	values = numpy.array([x.mean() for x in values]).flatten()

	lineplot(
		values   = [values],
		labels   = [mode.title()],
		title    = title,
		xlabel   = 'Epoch',
		ylabel   = 'Loss',
		filename = filename + '-' + metric
	)

def show_loss (report : Dict[str, Dict], title : str = None, filename : str = None) -> None :
	"""
	Doc
	"""

	train_loss = report['train']['loss']
	valid_loss = report['valid']['loss']

	lineplot(
		values   = [train_loss, valid_loss],
		labels   = ['Train', 'Valid'],
		title    = title,
		xlabel   = 'Epoch',
		ylabel   = 'Loss',
		filename = filename + '-loss'
	)

def show_r2 (report : Dict[str, Dict], title : str = None, filename : str = None) -> None :
	"""
	Doc
	"""

	train_r2 = report['train']['metric']['r2']
	valid_r2 = report['valid']['metric']['r2']

	train_r2 = numpy.array([r2.mean() for r2 in train_r2])
	valid_r2 = numpy.array([r2.mean() for r2 in valid_r2])

	lineplot(
		values   = [train_r2, valid_r2],
		labels   = ['Train', 'Valid'],
		title    = title,
		xlabel   = 'Epoch',
		ylabel   = 'R2 Score',
		filename = filename + '-r2'
	)

def show_accuracy (report : Dict[str, Dict], title : str = None, filename : str = None) :
	"""
	Doc
	"""

	train_acc = report['train']['metric']['acc']
	valid_acc = report['valid']['metric']['acc']

	train_acc = numpy.array([acc.mean() for acc in train_acc])
	valid_acc = numpy.array([acc.mean() for acc in valid_acc])

	lineplot(
		values   = [train_acc, valid_acc],
		labels   = ['Train', 'Valid'],
		title    = title,
		xlabel   = 'Epoch',
		ylabel   = 'Accuracy',
		filename = filename + '-accuracy'
	)

def show_lr (report : Dict[str, Dict], title : str = None, filename : str = None) :
	"""
	Doc
	"""

	train_lr = report['train']['lr']
	valid_lr = report['valid']['lr']

	lineplot(
		values   = [train_lr, valid_lr],
		labels   = ['Train', 'Valid'],
		title    = title,
		xlabel   = 'Epoch',
		ylabel   = 'Learning Rate',
		filename = filename + '-lr'
	)
