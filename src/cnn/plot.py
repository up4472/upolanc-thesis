from typing import Dict
from typing import List
from typing import Tuple

import math
import matplotlib
import numpy
import scipy
import seaborn

def compute_gridsize (n : int) -> Tuple[int, int, int] :
	"""
	Doc
	"""

	if n == 1 : return 1, 1, 1
	if n == 2 : return 2, 1, 2
	if n == 3 : return 3, 1, 3

	nrows = math.ceil(math.sqrt(n))
	ncols = math.ceil(n / nrows)

	return n, nrows, ncols

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

def show_prediction_error_grid (report : Dict[str, Dict], order : List[str], filename : str = None) -> None :
	"""
	Doc
	"""

	data = report['eval']['ypred'] - report['eval']['ytrue']

	n, nrows, ncols = compute_gridsize(
		n = numpy.shape(data)[1]
	)

	kwargs = {
		'sharex' : True,
		'sharey' : True,
		'figsize' : (16 * ncols, 10 * nrows)
	}

	if ncols > 1 :
		_, ax = matplotlib.pyplot.subplots(nrows, ncols, **kwargs)
	else :
		_, ax = matplotlib.pyplot.subplots(nrows, **kwargs)

	for index in range(n) :
		if nrows == 1 or ncols == 1 :
			axis = ax[index]
		else :
			axis = ax[index // ncols, index % ncols]

		seaborn.histplot(
			x     = data[:, index],
			ax    = axis,
			alpha = 0.9
		)

		axis.axvline(x = 0, color = 'r', linewidth = 3)
		axis.set_title(order[index].title())
		axis.set_xlabel('Prediction Error')

	for index in range(n, nrows * ncols) :
		if nrows == 1 or ncols == 1 :
			axis = ax[index]
		else :
			axis = ax[index // ncols, index % ncols]

		axis.axis('off')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-prediction_error.png',
			dpi    = 120,
			format = 'png'
		)

def show_prediction_error (report : Dict[str, Dict], order : List[str], group : str, filename : str = None) -> None :
	"""
	Doc
	"""

	data = report['eval']['ypred'] - report['eval']['ytrue']

	index = order.index(group)
	data = data[:, index]

	_, axis = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.histplot(
		x     = data,
		ax    = axis,
		alpha = 0.9
	)

	axis.axvline(x = 0, color = 'r', linewidth = 3)
	axis.set_title(group.title())
	axis.set_xlabel('Prediction Error')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-' + group + '-error.png',
			dpi    = 120,
			format = 'png'
		)
def show_linear_regression_grid (report : Dict[str, Dict], order : List[str], filename : str = None) -> None :
	"""
	Doc
	"""

	ypred = report['eval']['ypred']
	ytrue = report['eval']['ytrue']

	n, nrows, ncols = compute_gridsize(
		n = numpy.shape(ytrue)[1]
	)

	kwargs = {
		'sharex' : True,
		'sharey' : True,
		'figsize' : (10 * ncols, 10 * nrows)
	}

	for index in range(n) :
		variance = numpy.var(ypred[:, index], axis = None)

		if variance < 1e-7 :
			print('Variance in [{}] is almost zero ({:.2e}) : {:.8f}'.format(order[index], 1e-7, variance))
			return

	if ncols > 1 :
		_, ax = matplotlib.pyplot.subplots(nrows, ncols, **kwargs)
	else :
		_, ax = matplotlib.pyplot.subplots(nrows, **kwargs)

	for index in range(n) :
		if nrows == 1 or ncols == 1 :
			axis = ax[index]
		else :
			axis = ax[index // ncols, index % ncols]

		x1 = ypred[:, index]
		x2 = ytrue[:, index]

		seaborn.scatterplot(
			x     = x1,
			y     = x2,
			ax    = axis,
			alpha = 0.9
		)

		res = scipy.stats.linregress(x1, x2)

		axis.plot(x1, res.intercept + res.slope * x1,
			color     = 'r',
			linewidth = 2
		)

		xmin, xmax = axis.get_xlim()
		ymin, ymax = axis.get_ylim()

		gmin = min(xmin, ymin)
		gmax = max(xmax, ymax)

		axis.set_xlim([gmin, gmax])
		axis.set_ylim([gmin, gmax])

		axis.set_title(f'{order[index].title()} : k = {res.slope:.3f}; r = {res.rvalue:.3f}; p = {res.pvalue:.3e}')
		axis.set_aspect('equal')

	for index in range(n, nrows * ncols) :
		if nrows == 1 or ncols == 1 :
			axis = ax[index]
		else :
			axis = ax[index // ncols, index % ncols]

		axis.axis('off')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-prediction_linefit.png',
			dpi    = 120,
			format = 'png'
		)

def show_linear_regression (report : Dict[str, Dict], order : List[str], group : str, filename : str = None) -> None :
	"""
	Doc
	"""

	index = order.index(group)

	ypred = report['eval']['ypred'][:, index]
	ytrue = report['eval']['ytrue'][:, index]

	variance = numpy.var(ypred, axis = None)

	if variance < 1e-7 :
		print('Variance in [{}] is almost zero ({:.2e}) : {:.8f}'.format(group, 1e-7, variance))
		return

	_, axis = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.scatterplot(
		x  = ypred,
		y  = ytrue,
		ax = axis
	)

	res = scipy.stats.linregress(ypred, ytrue)

	axis.plot(
		ypred,
		res.intercept + res.slope * ypred,
		color = 'r',
		linewidth = 1
	)

	xmin, xmax = axis.get_xlim()
	ymin, ymax = axis.get_ylim()

	gmin = min(xmin, ymin)
	gmax = max(xmax, ymax)

	axis.set_xlim([gmin, gmax])
	axis.set_ylim([gmin, gmax])

	axis.set_aspect('equal')

	axis.set_title(f'k = {res.slope:.3f}; r = {res.rvalue:.3f}; p = {res.pvalue:.3e}')
	axis.set_xlabel('Predicted')
	axis.set_ylabel('Actual')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-' + group + '-linefit.png',
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
			case 'mse'        : names.append('MSE')
			case 'mae'        : names.append('MAE')
			case 'smooth-mae' : names.append('Smooth MAE')
			case 'huber'      : names.append('Huber')
			case 'r2'         : names.append('R2 Score')
			case 'entropy'    : names.append('Cross Entropy')
			case 'nll'        : names.append('Negative Log Likelihood')
			case 'accuracy'   : names.append('Accuracy')
			case _            : names.append('?')

	n, nrows, ncols = compute_gridsize(
		n = len(names)
	)

	kwargs = {
		'figsize' : (16 * ncols, 10 * nrows)
	}

	if ncols > 1 :
		_, ax = matplotlib.pyplot.subplots(nrows, ncols, **kwargs)
	else :
		_, ax = matplotlib.pyplot.subplots(nrows, **kwargs)

	for index, (metric, name) in enumerate(zip(metrics, names)):
		metric = report[mode]['metric'][metric]
		metric = numpy.array([x.mean() for x in metric])

		if nrows == 1 or ncols == 1 :
			axis = ax[index]
		else :
			axis = ax[index // ncols, index % ncols]

		seaborn.lineplot(x = numpy.arange(1, 1 + len(metric)), y = metric, ax = axis)

		axis.set_xlabel('Epoch')
		axis.set_ylabel('Loss')
		axis.set_title(name)

	for index in range(n, nrows * ncols) :
		if nrows == 1 or ncols == 1 :
			axis = ax[index]
		else :
			axis = ax[index // ncols, index % ncols]

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

	train_acc = report['train']['metric']['accuracy']
	valid_acc = report['valid']['metric']['accuracy']

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
