from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import math
import matplotlib
import numpy
import scipy
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay

def get_regression_limits (report : Dict[str, Dict]) -> Dict[str, Tuple] :
	"""
	Doc
	"""

	limits = {
		'r2'    : (-1.0, 1.0),
		'mae'   : ( 0.0, 5.0),
		'mse'   : ( 0.0, 5.0),
		'mape'  : ( 0.0, 2.0),
		'wmape' : ( 0.0, 2.0),
		'loss'  : ( 0.0, 5.0)
	}

	for key in limits.keys() :
		if key == 'r2' : continue

		if key in report['metric'].keys() :
			bot = numpy.nanmin(report['metric'][key], axis = None)
			top = numpy.nanmax(report['metric'][key], axis = None)

			limits[key] = (
				max(limits[key][0], bot),
				min(limits[key][1], top),
			)

	return limits

def get_classification_limits (report : Dict[str, Dict]) -> Dict[str, Tuple] :
	"""
	Doc
	"""

	limits = {
		'acc'   : ( 0.0,  1.0),
		'loss'  : ( 0.0, 25.0)
	}

	for key in limits.keys() :
		if key in report.keys() :
			limits[key] = (
				max(limits[key], report['metric'][key].nanmin()),
				min(limits[key], report['metric'][key].nanmax()),
			)

	return limits

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

def lineplot (values : List[Union[List, numpy.ndarray]], labels : List[str], xlabel : str, ylabel : str, title : str = None, filename : str = None, limit_bot : float = None, limit_top : float = None, start_index : int = None) -> None :
	"""
	Doc
	"""

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	if start_index is None :
		start_index = 0
	elif start_index >= len(values[0]) :
		start_index = 0

	x = numpy.arange(1, 1 + len(values[0]))

	for y, label in zip(values, labels) :
		seaborn.lineplot(
			x     = x[start_index:],
			y     = y[start_index:],
			label = label,
			ax    = ax
		)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	if title is not None :
		ax.set_title(title)

	ax.set_ylim(limit_bot, limit_top)

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

	if len(order) == 1 :
		show_prediction_error(
			report   = report,
			order    = order,
			group    = order[0],
			filename = filename
		)

		return

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

	if len(order) == 1 :
		show_linear_regression(
			report   = report,
			order    = order,
			group    = order[0],
			filename = filename
		)

		return

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

def show_metric_grid (report : Dict[str, Dict], mode : str = 'train', filename : str = None, apply_limits : bool = False, start_index : int = None) -> None :
	"""
	Doc
	"""

	metrics = list(report[mode]['metric'].keys())
	names   = list()
	limits  = None

	if start_index is None :
		start_index = 0

	if apply_limits :
		if 'r2' in report[mode]['metric'].keys() :
			limits = get_regression_limits(report = report[mode])

		if 'accuracy' in report[mode]['metric'].keys() :
			limits = None

	for metric in metrics :
		metric = metric.lower()

		if   metric == 'accuracy'   : names.append(['Score',  'Accuracy'])
		elif metric == 'ap'         : names.append(['Score',  'Average Precision'])
		elif metric == 'auroc'      : names.append(['Score',  'AUROC'])
		elif metric == 'bce'        : names.append(['Loss',   'Binary Cross Entropy'])
		elif metric == 'entropy'    : names.append(['Loss',   'Cross Entropy'])
		elif metric == 'f1'         : names.append(['Score',  'F1 Score'])
		elif metric == 'huber'      : names.append(['Loss',   'Huber'])
		elif metric == 'jaccardi'   : names.append(['Score',  'Jaccardi Index'])
		elif metric == 'mae'        : names.append(['Loss',   'MAE'])
		elif metric == 'mape'       : names.append(['Loss',   'MAPE'])
		elif metric == 'matthews'   : names.append(['Score',  'Matthews Correlation Coef.'])
		elif metric == 'mse'        : names.append(['Loss',   'MSE'])
		elif metric == 'nll'        : names.append(['Loss',   'Negative Log Likelihood'])
		elif metric == 'r2'         : names.append(['Score',  'R2 Score'])
		elif metric == 'smae'       : names.append(['Loss',   'Smooth MAE'])
		elif metric == 'wmape'      : names.append(['Loss',   'Weighted MAPE'])
		else                        : pass

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
		if apply_limits and limits is not None and metric in limits.keys() :
			ylimit = limits[metric]
		else :
			ylimit = None

		metric = report[mode]['metric'][metric]
		metric = numpy.array([x.mean() for x in metric])

		if nrows == 1 or ncols == 1 :
			axis = ax[index]
		else :
			axis = ax[index // ncols, index % ncols]

		x = numpy.arange(1, 1 + len(metric))
		y = metric

		if start_index >= len(metric) :
			start_index = 0

		seaborn.lineplot(
			x   = x[start_index:],
			y   = y[start_index:],
			ax  = axis
		)

		axis.set_xlabel('Epoch')
		axis.set_ylabel(name[0])
		axis.set_title(name[1])

		if ylimit is not None :
			axis.set_ylim(
				ylimit[0],
				ylimit[1]
			)

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

def show_metric (report : Dict[str, Dict], mode : str, metric : str, title : str = None, filename : str = None, limit_bot : float = None, limit_top : float = None, start_index : int = None) -> None :
	"""
	Doc
	"""

	values = report[mode]['metric'][metric]
	values = numpy.array([x.mean() for x in values]).flatten()

	lineplot(
		values      = [values],
		labels      = [mode.title()],
		title       = title,
		xlabel      = 'Epoch',
		ylabel      = 'Loss',
		filename    = filename + '-' + metric,
		limit_bot   = limit_bot,
		limit_top   = limit_top,
		start_index = start_index
	)

def show_loss (report : Dict[str, Dict], title : str = None, filename : str = None, limit_bot : float = None, limit_top : float = None, start_index : int = None) -> None :
	"""
	Doc
	"""

	train_loss = report['train']['loss']
	valid_loss = report['valid']['loss']

	lineplot(
		values      = [train_loss, valid_loss],
		labels      = ['Train', 'Valid'],
		title       = title,
		xlabel      = 'Epoch',
		ylabel      = 'Loss',
		filename    = filename + '-loss',
		limit_bot   = limit_bot,
		limit_top   = limit_top,
		start_index = start_index
	)

def show_r2 (report : Dict[str, Dict], title : str = None, filename : str = None, limit_bot : float = None, limit_top : float = None, start_index : int = None) -> None :
	"""
	Doc
	"""

	train_r2 = report['train']['metric']['r2']
	valid_r2 = report['valid']['metric']['r2']

	train_r2 = numpy.array([r2.mean() for r2 in train_r2])
	valid_r2 = numpy.array([r2.mean() for r2 in valid_r2])

	lineplot(
		values      = [train_r2, valid_r2],
		labels      = ['Train', 'Valid'],
		title       = title,
		xlabel      = 'Epoch',
		ylabel      = 'R2 Score',
		filename    = filename + '-r2',
		limit_bot   = limit_bot,
		limit_top   = limit_top,
		start_index = start_index
	)

def show_accuracy (report : Dict[str, Dict], title : str = None, filename : str = None, limit_bot : float = None, limit_top : float = None, start_index : int = None) -> None :
	"""
	Doc
	"""

	train_acc = report['train']['metric']['accuracy']
	valid_acc = report['valid']['metric']['accuracy']

	train_acc = numpy.array([acc.mean() for acc in train_acc])
	valid_acc = numpy.array([acc.mean() for acc in valid_acc])

	lineplot(
		values      = [train_acc, valid_acc],
		labels      = ['Train', 'Valid'],
		title       = title,
		xlabel      = 'Epoch',
		ylabel      = 'Accuracy',
		filename    = filename + '-accuracy',
		limit_bot   = limit_bot,
		limit_top   = limit_top,
		start_index = start_index
	)

def show_lr (report : Dict[str, Dict], title : str = None, filename : str = None, limit_bot : float = None, limit_top : float = None, start_index : int = None) -> None :
	"""
	Doc
	"""

	train_lr = report['train']['lr']
	valid_lr = report['valid']['lr']

	lineplot(
		values      = [train_lr, valid_lr],
		labels      = ['Train', 'Valid'],
		title       = title,
		xlabel      = 'Epoch',
		ylabel      = 'Learning Rate',
		filename    = filename + '-lr',
		limit_bot   = limit_bot,
		limit_top   = limit_top,
		start_index = start_index
	)

def plot_confusion_matrix (report : Dict[str, Dict], filename : str = None) -> None :
	"""
	Doc
	"""

	matrix = report['eval']['metric']['confusion']

	if numpy.ndim(matrix) == 2 :
		d0 = numpy.size(matrix, axis = 0) // 2
		d1 = numpy.size(matrix, axis = 1)
		d2 = 2

		matrix = matrix.reshape((d0, d1, d2))

	matrix = matrix.sum(axis = 0)

	matrix = ConfusionMatrixDisplay(
		confusion_matrix = matrix,
		display_labels   = [False, True]
	)

	matrix.plot()

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-confusion.png',
			dpi    = 120,
			format = 'png'
		)
