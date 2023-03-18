from pandas import DataFrame
from typing import Tuple

import matplotlib
import os
import seaborn

from source.python.io.loader import load_csv

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
		filename   = filename + '-loss'
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
		filename   = filename + '-r2'
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
		filename  = filename + '-loss'
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
		filename  = filename + '-r2'
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
		filename  = filename + '-lr'
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
		filename  = filename + '-lambda-loss'
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
		filename  = filename + '-lambda-r2'
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
		filename  = filename + '-lambda-mape'
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
		filename  = filename + '-lambda-wmape'
	)
