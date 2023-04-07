from pandas import DataFrame
from typing import Tuple
from typing import Optional

import matplotlib
import os
import seaborn

from source.python.io.loader import load_csv

def format_float_tick (label : str) -> str :
	"""
	Doc
	"""

	if len(label) >= 1 and label[0] != '-' :
		label = ' ' + label

	while len(label) < 8 :
		label = label + '0'

	return '{:8s}'.format(label[:8])

def trials_lineplot (dataframe : DataFrame, y : str, ylabel : str, ascending : bool, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	dataframe = dataframe.copy()
	dataframe = dataframe.sort_values(y, ascending = ascending)

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	for iteration, directory in zip(dataframe['training_iteration'], dataframe['logdir']) :
		progress = load_csv(
			filename = os.path.join(directory, 'progress.csv')
		)

		seaborn.lineplot(
			data  = progress,
			x     = 'training_iteration',
			y     = y,
			ax    = ax,
			alpha = alpha,
			label = str(progress['trial_id'].iloc[0])
		)

		max_trials = max_trials - 1

		if max_trials <= 0 :
			break

	ax.set_xlabel('Epoch')
	ax.set_ylabel(ylabel)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format = 'png',
			dpi    = 120
		)


def trials_lineplot_loss (dataframe : DataFrame, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	trials_lineplot(
		dataframe = dataframe,
		y          = 'valid_loss',
		ylabel     = 'Valid Loss',
		ascending  = True,
		max_trials = max_trials,
		alpha      = alpha,
		filename   = filename + '-line-loss'
	)

def trials_lineplot_accuracy (dataframe : DataFrame, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	trials_lineplot(
		dataframe = dataframe,
		y          = 'valid_accuracy',
		ylabel     = 'Valid Accuracy',
		ascending  = False,
		max_trials = max_trials,
		alpha      = alpha,
		filename   = filename + '-line-accuracy'
	)

def trials_lineplot_r2 (dataframe : DataFrame, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	trials_lineplot(
		dataframe = dataframe,
		y          = 'valid_r2',
		ylabel     = 'Valid R2',
		ascending  = False,
		max_trials = max_trials,
		alpha      = alpha,
		filename   = filename + '-line-r2'
	)

def trial_lineplot (dataframe : DataFrame, y : str, ylabel : str, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.lineplot(
		data  = dataframe,
		x     = 'training_iteration',
		y     = y,
		ax    = ax,
		alpha = alpha,
		color = color,
		label = str(dataframe['trial_id'].iloc[0])
	)

	ycomp = None

	if y == 'valid_loss' : ycomp = 'train_loss'
	if y == 'train_loss' : ycomp = 'valid_loss'

	if ycomp is not None :
		seaborn.lineplot(
			data   = dataframe,
			x      = 'training_iteration',
			y      = ycomp,
			ax     = ax,
			alpha  = 0.5,
			color  = 'k',
			label  = str(dataframe['trial_id'].iloc[0]) + '-' + ycomp.split('_')[0]
		)

	ax.set_xlabel('Epoch')
	ax.set_ylabel(ylabel)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format = 'png',
			dpi    = 120
		)

def trial_lineplot_loss (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	trial_lineplot(
		dataframe = dataframe,
		y         = 'valid_loss',
		ylabel    = 'Valid Loss',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-line-loss'
	)

def trial_lineplot_accuracy (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	trial_lineplot(
		dataframe = dataframe,
		y         = 'valid_accuracy',
		ylabel    = 'Valid Accuracy',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-line-accuracy'
	)

def trial_lineplot_r2 (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	trial_lineplot(
		dataframe = dataframe,
		y         = 'valid_r2',
		ylabel    = 'Valid R2',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-line-r2'
	)

def trial_lineplot_lr (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	trial_lineplot(
		dataframe = dataframe,
		y         = 'lr',
		ylabel    = 'Learning Rate',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-line-lr'
	)

def trials_scatterplot (dataframe : DataFrame, x : str, y : str, xlabel : str, ylabel : str, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.scatterplot(
		data  = dataframe,
		x     = x,
		y     = y,
		ax    = ax,
		alpha = alpha,
		color = color,
	)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format = 'png',
			dpi    = 120
		)

def trials_scatterplot_lambda_loss (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None, clip : Tuple[int, int] = None) -> None :
	"""
	Doc
	"""

	if clip is not None :
		dataframe = dataframe.copy()
		dataframe['valid_loss'] = dataframe['valid_loss'].clip(
			lower = clip[0],
			upper = clip[1]
		)

	trials_scatterplot(
		dataframe = dataframe,
		x         = 'config/boxcox/lambda',
		y         = 'valid_loss',
		xlabel    = 'Lambda',
		ylabel    = 'Valid Loss',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-scatter-lambda-loss'
	)

def trials_scatterplot_lambda_r2 (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None, clip : Tuple[int, int] = None) -> None :
	"""
	Doc
	"""

	if clip is not None :
		dataframe = dataframe.copy()
		dataframe['valid_r2'] = dataframe['valid_r2'].clip(
			lower = clip[0],
			upper = clip[1]
		)

	trials_scatterplot(
		dataframe = dataframe,
		x         = 'config/boxcox/lambda',
		y         = 'valid_r2',
		xlabel    = 'Lambda',
		ylabel    = 'Valid R2',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-scatter-lambda-r2'
	)

def trials_scatterplot_lambda_mape (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None, clip : Tuple[int, int] = None) -> None :
	"""
	Doc
	"""

	if clip is not None :
		dataframe = dataframe.copy()
		dataframe['valid_mape'] = dataframe['valid_mape'].clip(
			lower = clip[0],
			upper = clip[1]
		)

	trials_scatterplot(
		dataframe = dataframe,
		x         = 'config/boxcox/lambda',
		y         = 'valid_mape',
		xlabel    = 'Lambda',
		ylabel    = 'Valid MAPE',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-scatter-lambda-mape'
	)

def trials_scatterplot_lambda_wmape (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None, clip : Tuple[int, int] = None) -> None :
	"""
	Doc
	"""

	if clip is not None :
		dataframe = dataframe.copy()
		dataframe['valid_wmape'] = dataframe['valid_wmape'].clip(
			lower = clip[0],
			upper = clip[1]
		)

	trials_scatterplot(
		dataframe = dataframe,
		x         = 'config/boxcox/lambda',
		y         = 'valid_wmape',
		xlabel    = 'Lambda',
		ylabel    = 'Valid WMAPE',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-scatter-lambda-wmape'
	)

def trials_scatterplot_lambda_accuracy (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None, clip : Tuple[int, int] = None) -> None :
	"""
	Doc
	"""

	if clip is not None :
		dataframe = dataframe.copy()
		dataframe['valid_accuracy'] = dataframe['valid_accuracy'].clip(
			lower = clip[0],
			upper = clip[1]
		)

	trials_scatterplot(
		dataframe = dataframe,
		x         = 'config/boxcox/lambda',
		y         = 'valid_accuracy',
		xlabel    = 'Lambda',
		ylabel    = 'Valid Accuracy',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-scatter-lambda-accuracy'
	)

def trials_scatterplot_lambda_auroc (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None, clip : Tuple[int, int] = None) -> None :
	"""
	Doc
	"""

	if clip is not None :
		dataframe = dataframe.copy()
		dataframe['valid_auroc'] = dataframe['valid_auroc'].clip(
			lower = clip[0],
			upper = clip[1]
		)

	trials_scatterplot(
		dataframe = dataframe,
		x         = 'config/boxcox/lambda',
		y         = 'valid_auroc',
		xlabel    = 'Lambda',
		ylabel    = 'Valid AUROC',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-scatter-lambda-auroc'
	)

def trials_scatterplot_lambda_f1 (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None, clip : Tuple[int, int] = None) -> None :
	"""
	Doc
	"""

	if clip is not None :
		dataframe = dataframe.copy()
		dataframe['valid_f1'] = dataframe['valid_f1'].clip(
			lower = clip[0],
			upper = clip[1]
		)

	trials_scatterplot(
		dataframe = dataframe,
		x         = 'config/boxcox/lambda',
		y         = 'valid_f1',
		xlabel    = 'Lambda',
		ylabel    = 'Valid F1',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-scatter-lambda-f1'
	)

def trials_heatmap_lambda_bins (dataframe : DataFrame, vmin : Optional[float], vmax : Optional[float], values : str = 'valid_accuracy', filename : str = None) -> None :
	"""
	Doc
	"""

	data = dataframe.pivot(
		index   = 'config/boxcox/lambda',
		columns = 'config/class/bins',
		values  = values
	)

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.heatmap(
		data      = data,
		cmap      = 'crest',
		annot     = False,
		linewidth = 0.5,
		vmin      = vmin,
		vmax      = vmax,
		square    = True,
		ax        = ax
	)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format = 'png',
			dpi    = 120
		)

def trials_heatmap_lambda_bins_accuracy (dataframe : DataFrame, filename : str = None) -> None :
	"""
	Doc
	"""

	trials_heatmap_lambda_bins(
		dataframe = dataframe,
		values    = 'valid_accuracy',
		vmin      = 0.0,
		vmax      = 1.0,
		filename  = filename + '-heatmap-lambda-bins-accuracy'
	)

def trials_heatmap_lambda_bins_f1 (dataframe : DataFrame, filename : str = None) -> None :
	"""
	Doc
	"""

	trials_heatmap_lambda_bins(
		dataframe = dataframe,
		values    = 'valid_f1',
		vmin      = None,
		vmax      = None,
		filename  = filename + '-heatmap-lambda-bins-f1'
	)

def trials_heatmap_lambda_bins_auroc (dataframe : DataFrame, filename : str = None) -> None :
	"""
	Doc
	"""

	trials_heatmap_lambda_bins(
		dataframe = dataframe,
		values    = 'valid_auroc',
		vmin      = None,
		vmax      = None,
		filename  = filename + '-heatmap-lambda-bins-auroc'
	)

def trials_heatmap_lambda_bins_matthews (dataframe : DataFrame, filename : str = None) -> None :
	"""
	Doc
	"""

	trials_heatmap_lambda_bins(
		dataframe = dataframe,
		values    = 'valid_matthews',
		vmin      = None,
		vmax      = None,
		filename  = filename + '-heatmap-lambda-bins-matthews'
	)
