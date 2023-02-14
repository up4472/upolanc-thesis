from pandas                   import DataFrame
from torch.nn                 import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing                   import Any
from typing                   import Dict

from ray import tune

import matplotlib
import numpy
import os
import seaborn
import torch

from source.python.cnn.models  import Washburn2019r
from source.python.cnn.models  import Zrimec2020r

from source.python.cnn._common import evaluate_epoch
from source.python.cnn._common import train_epoch
from source.python.cnn.core    import lock_random
from source.python.cnn.dataset import to_dataloaders
from source.python.cnn.model   import get_criterion
from source.python.cnn.model   import get_model_trainers
from source.python.cnn.model   import he_uniform_weight
from source.python.cnn.model   import zero_bias
from source.python.io.loader   import load_csv

def get_tune_model (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> Module :
	if core_config['model/type'] == 'zrimec2020r' :
		model = Zrimec2020r(params = tune_config | {
			'model/input/channels' : core_config['model/input/channels'],
			'model/input/height'   : core_config['model/input/height'],
			'model/input/width'    : core_config['model/input/width'],
			'model/input/features' : core_config['model/input/features'],
			'model/fc3/features'   : core_config['model/output/size']
		})

	elif core_config['model/type'] == 'washburn2019r' :
		model = Washburn2019r(params = tune_config | {
			'model/input/channels' : core_config['model/input/channels'],
			'model/input/height'   : core_config['model/input/height'],
			'model/input/width'    : core_config['model/input/width'],
			'model/input/features' : core_config['model/input/features'],
			'model/fc3/features'   : core_config['model/output/size']
		})

	else :
		raise ValueError()

	model = model.double()
	model = model.apply(he_uniform_weight)
	model = model.apply(zero_bias)
	model = model.to(core_config['core/device'])

	return model

def regression_tune (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	lock_random(seed = core_config['core/random'])

	dataloaders = to_dataloaders(
		dataset     = torch.load(core_config['dataset/filepath']),
		random_seed = core_config['core/random'],
		split_size  = {
			'valid' : core_config['dataset/split/valid'],
			'test'  : core_config['dataset/split/test']
		},
		batch_size  = {
			'train' : tune_config['dataset/batch_size'],
			'valid' : tune_config['dataset/batch_size'],
			'test'  : tune_config['dataset/batch_size']
		}
	)

	train_dataloader = dataloaders[0]
	valid_dataloader = dataloaders[1]
	test_dataloader  = dataloaders[2]

	model = get_tune_model(
		tune_config = tune_config,
		core_config = core_config
	)

	model_trainers = get_model_trainers(
		model  = model,
		config = tune_config,
		epochs = core_config['model/epochs']
	)

	criterion = model_trainers['criterion']
	optimizer = model_trainers['optimizer']
	scheduler = model_trainers['scheduler']

	model_params = {
		'model'     : model,
		'criterion' : criterion,
		'optimizer' : optimizer,
		'scheduler' : scheduler,
		'device'    : core_config['core/device'],
		'verbose'   : False,
		'train_dataloader' : train_dataloader,
		'valid_dataloader' : valid_dataloader,
		'test_dataloader'  : test_dataloader,
		'metrics' : {
			'r2'  : get_criterion(reduction = 'mean', weights = None, query = 'r2'),
			'mae' : get_criterion(reduction = 'mean', weights = None, query = 'mae')
		}
	}

	for epoch in range(core_config['model/epochs']) :
		current_lr = optimizer.param_groups[0]['lr']

		train_report = train_epoch(model = model, params = model_params, desc = '')
		valid_report = evaluate_epoch(model = model, params = model_params, desc = '', validation = True)

		train_loss = train_report['loss']
		valid_loss = valid_report['loss']
		valid_r2   = valid_report['metric']['r2']
		valid_mae  = valid_report['metric']['mae']

		if scheduler is not None :
			if isinstance(scheduler, ReduceLROnPlateau) :
				scheduler.step(valid_loss)
			else :
				scheduler.step()

		if core_config['tuner/checkpoint'] :
			with tune.checkpoint_dir(epoch) as checkpoint :
				path = os.path.join(checkpoint, 'checkpoint')
				data = (
					model.state_dict(),
					optimizer.state_dict()
				)

				torch.save(data, path)

		tune.report(
			valid_loss = valid_loss,
			valid_r2   = numpy.mean(valid_r2),
			valid_mae  = numpy.mean(valid_mae),
			train_loss = train_loss,
			lr         = current_lr
		)

def plot_trials (dataframe : DataFrame, y : str, ylabel : str, ascending : bool, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
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
			label = progress['trial_id'].iloc[0]
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

def plot_trials_loss (dataframe : DataFrame, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	plot_trials(
		dataframe = dataframe,
		y          = 'valid_loss',
		ylabel     = 'Valid Loss',
		ascending  = True,
		max_trials = max_trials,
		alpha      = alpha,
		filename   = filename + '-loss'
	)

def plot_trials_r2 (dataframe : DataFrame, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	plot_trials(
		dataframe = dataframe,
		y          = 'valid_r2',
		ylabel     = 'Valid R2',
		ascending  = False,
		max_trials = max_trials,
		alpha      = alpha,
		filename   = filename + '-r2'
	)

def plot_trial (dataframe : DataFrame, y : str, ylabel : str, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
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
		label = dataframe['trial_id'].iloc[0]
	)

	ax.set_xlabel('Epoch')
	ax.set_ylabel(ylabel)

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format = 'png',
			dpi    = 120
		)

def plot_trial_loss (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	plot_trial(
		dataframe = dataframe,
		y         = 'valid_loss',
		ylabel    = 'Valid Loss',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-loss'
	)

def plot_trial_r2 (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	plot_trial(
		dataframe = dataframe,
		y         = 'valid_r2',
		ylabel    = 'R2',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-r2'
	)

def plot_trial_lr (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	plot_trial(
		dataframe = dataframe,
		y         = 'lr',
		ylabel    = 'Learning Rate',
		alpha     = alpha,
		color     = color,
		filename  = filename + '-lr'
	)
