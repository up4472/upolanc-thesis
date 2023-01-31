from pandas   import DataFrame
from torch.nn import Module
from typing   import Any
from typing   import Dict

from ray import tune

import matplotlib
import numpy
import os
import seaborn
import torch

from src.cnn.dataset import to_dataloaders
from src.cnn.dataset import to_dataset
from src.cnn.models  import Washburn2019r
from src.cnn.models  import Zrimec2020r

from src.cnn._common import evaluate_epoch
from src.cnn._common import train_epoch
from src.cnn.core    import lock_random
from src.cnn.model   import get_criterion
from src.cnn.model   import get_optimizer
from src.cnn.model   import get_scheduler
from src.cnn.model   import he_uniform_weight
from src.cnn.model   import zero_bias
from src.io.loader   import load_csv

def get_washburn2019r (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> Module :
	"""
	Doc
	"""

	model = Washburn2019r(params = {
		'other' : {
			'in_height'   : core_config['input']['height'],
			'in_width'    : core_config['input']['width'],
			'in_features' : core_config['input']['features'],
			'dropout'     : tune_config['model/dropout']
		},
		'conv1' : {
			'filters' : tune_config['model/conv1/filters'],
			'kernel'  : tune_config['model/conv1/kernel'],
			'padding' : 'none'
		},
		'conv2' : {
			'filters' : tune_config['model/conv2/filters'],
			'kernel'  : tune_config['model/conv2/kernel'],
			'padding' : tune_config['model/conv2/padding']
		},
		'conv3' : {
			'filters' : tune_config['model/conv3/filters'],
			'kernel'  : tune_config['model/conv3/kernel'],
			'padding' : tune_config['model/conv3/padding']
		},
		'conv4' : {
			'filters' : tune_config['model/conv4/filters'],
			'kernel'  : tune_config['model/conv4/kernel'],
			'padding' : tune_config['model/conv4/padding']
		},
		'conv5' : {
			'filters' : tune_config['model/conv5/filters'],
			'kernel'  : tune_config['model/conv5/kernel'],
			'padding' : tune_config['model/conv5/padding']
		},
		'conv6' : {
			'filters' : tune_config['model/conv6/filters'],
			'kernel'  : tune_config['model/conv6/kernel'],
			'padding' : tune_config['model/conv6/padding']
		},
		'maxpool1' : {
			'kernel'  : tune_config['model/maxpool1/kernel'],
			'padding' : tune_config['model/maxpool1/padding']
		},
		'maxpool2' : {
			'kernel'  : tune_config['model/maxpool2/kernel'],
			'padding' : tune_config['model/maxpool2/padding']
		},
		'maxpool3' : {
			'kernel'  : tune_config['model/maxpool3/kernel'],
			'padding' : tune_config['model/maxpool3/padding']
		},
		'fc1' : {
			'features' : tune_config['model/fc1/features']
		},
		'fc2' : {
			'features' : tune_config['model/fc2/features']
		},
		'fc3' : {
			'features' : core_config['output']['length']
		}
	})

	model = model.double()
	model = model.apply(he_uniform_weight)
	model = model.apply(zero_bias)
	model = model.to(core_config['device'])

	return model

def get_zrimec2020r (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> Module :
	"""
	Doc
	"""

	model = Zrimec2020r(params = {
		'other' : {
			'in_height'   : core_config['input']['height'],
			'in_width'    : core_config['input']['width'],
			'in_features' : core_config['input']['features'],
			'dropout'     : tune_config['model/dropout']
		},
		'conv1' : {
			'filters' : tune_config['model/conv1/filters'],
			'kernel'  : tune_config['model/conv1/kernel'],
			'padding' : 'none'
		},
		'conv2' : {
			'filters' : tune_config['model/conv2/filters'],
			'kernel'  : tune_config['model/conv2/kernel'],
			'padding' : tune_config['model/conv2/padding']
		},
		'conv3' : {
			'filters' : tune_config['model/conv2/filters'],
			'kernel'  : tune_config['model/conv2/kernel'],
			'padding' : tune_config['model/conv2/padding']
		},
		'maxpool1' : {
			'kernel'  : tune_config['model/maxpool1/kernel'],
			'padding' : tune_config['model/maxpool1/padding']
		},
		'maxpool2' : {
			'kernel'  : tune_config['model/maxpool2/kernel'],
			'padding' : tune_config['model/maxpool2/padding']
		},
		'maxpool3' : {
			'kernel'  : tune_config['model/maxpool3/kernel'],
			'padding' : tune_config['model/maxpool3/padding']
		},
		'fc1' : {
			'features' : tune_config['model/fc1/features']
		},
		'fc2' : {
			'features' : tune_config['model/fc2/features']
		},
		'fc3' : {
			'features' : core_config['output']['length']
		}
	})

	model = model.double()
	model = model.apply(he_uniform_weight)
	model = model.apply(zero_bias)
	model = model.to(core_config['device'])

	return model

def tune_method (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	lock_random(
		seed = core_config['random_seed']
	)

	gene_sequences = core_config['files']['sequences']()
	gene_frequency = core_config['files']['frequency']()

	values = core_config['files']['values']()

	gene_targets = {
		key : value[core_config['output']['group1']]
		for key, value in values.items()
	}

	dataset = to_dataset(
		sequences   = gene_sequences,
		features    = gene_frequency,
		targets     = gene_targets,
		expand_dims = core_config['expand_dims']
	)

	dataloaders = to_dataloaders(
		dataset     = dataset,
		split_size  = core_config['split_size'],
		batch_size  = {
			'train' : tune_config['dataset/batch_size'],
			'valid' : tune_config['dataset/batch_size'],
			'test'  : tune_config['dataset/batch_size']
		},
		random_seed = core_config['random_seed']
	)

	train_dataloader = dataloaders[0]
	valid_dataloader = dataloaders[1]
	test_dataloader  = dataloaders[2]

	match core_config['model_name'] :
		case 'zrimec2020r'   : model = get_zrimec2020r
		case 'washburn2019r' : model = get_washburn2019r
		case _ : raise ValueError()

	model = model(
		tune_config = tune_config,
		core_config = core_config
	)

	criterion = get_criterion(
		query     = 'mse',
		reduction = 'mean',
		weights   = None
	)

	optimizer = get_optimizer(
		query        = 'adam',
		model        = model,
		betas        = (
			tune_config['optimizer/momentum'],
			0.999
		),
		lr           = tune_config['optimizer/lr'],
		weight_decay = tune_config['optimizer/decay']
	)

	scheduler = get_scheduler(
		query     = 'plateau',
		mode      = 'min',
		optimizer = optimizer,
		patience  = 5,
		factor    = 0.5,
		min_lr    = 1e-8
	)

	params = {
		'model'     : model,
		'criterion' : criterion,
		'optimizer' : optimizer,
		'scheduler' : scheduler,
		'device'    : core_config['device'],
		'verbose'   : False,
		'train_dataloader' : train_dataloader,
		'valid_dataloader' : valid_dataloader,
		'test_dataloader'  : test_dataloader,
		'metrics' : {
			'r2'  : get_criterion(reduction = 'mean', weights = None, query = 'r2'),
			'mae' : get_criterion(reduction = 'mean', weights = None, query = 'mae')
		}
	}

	for epoch in range(core_config['epochs']) :
		train_report = train_epoch(model = model, params = params, desc = '')
		valid_report = evaluate_epoch(model = model, params = params, desc = '', validation = True)

		train_loss = train_report['loss']
		valid_loss = valid_report['loss']
		valid_r2   = valid_report['metric']['r2']
		valid_mae  = valid_report['metric']['mae']

		if params['scheduler'] is not None :
			params['scheduler'].step(valid_loss)

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
			train_loss = train_loss
		)

def plot_trials_loss (dataframe : DataFrame, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	dataframe = dataframe.copy()
	dataframe = dataframe.sort_values('valid_loss', ascending = True)

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	for trial_id, training_iteration, directory in zip(dataframe['trial_id'], dataframe['training_iteration'], dataframe['logdir']) :
		if training_iteration <= 1 :
			continue

		progress = load_csv(
			filename = os.path.join(directory, 'progress.csv')
		)

		seaborn.lineplot(
			data  = progress,
			x     = 'training_iteration',
			y     = 'valid_loss',
			ax    = ax,
			alpha = alpha,
			label = progress['trial_id'].iloc[0]
		)

		max_trials = max_trials - 1

		if max_trials <= 0 :
			break

	ax.set_xlabel('Epoch')
	ax.set_ylabel('Valid Loss')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-loss.png',
			format = 'png',
			dpi    = 120
		)

def plot_trials_r2 (dataframe : DataFrame, max_trials : int = 10, alpha : float = 0.9, filename : str = None) -> None :
	"""
	Doc
	"""

	dataframe = dataframe.copy()
	dataframe = dataframe.sort_values('valid_r2', ascending = False)

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	for trial_id, training_iteration, directory in zip(dataframe['trial_id'], dataframe['training_iteration'], dataframe['logdir']) :
		if training_iteration <= 1 :
			continue

		progress = load_csv(
			filename = os.path.join(directory, 'progress.csv')
		)

		seaborn.lineplot(
			data  = progress,
			x     = 'training_iteration',
			y     = 'valid_r2',
			ax    = ax,
			alpha = alpha,
			label = progress['trial_id'].iloc[0]
		)

		max_trials = max_trials - 1

		if max_trials <= 0 :
			break

	ax.set_xlabel('Epoch')
	ax.set_ylabel('Valid R2')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-r2.png',
			format = 'png',
			dpi    = 120
		)

def plot_trial_loss (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.lineplot(
		data  = dataframe,
		x     = 'training_iteration',
		y     = 'valid_loss',
		ax    = ax,
		alpha = alpha,
		color = color,
		label = dataframe['trial_id'].iloc[0]
	)

	ax.set_xlabel('Epoch')
	ax.set_ylabel('Valid Loss')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-loss.png',
			format = 'png',
			dpi    = 120
		)

def plot_trial_r2 (dataframe : DataFrame, alpha : float = 0.9, color : str = 'b', filename : str = None) -> None :
	"""
	Doc
	"""

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	seaborn.lineplot(
		data  = dataframe,
		x     = 'training_iteration',
		y     = 'valid_r2',
		ax    = ax,
		alpha = alpha,
		color = color,
		label = dataframe['trial_id'].iloc[0]
	)

	ax.set_xlabel('Epoch')
	ax.set_ylabel('Valid R2')

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '-r2.png',
			format = 'png',
			dpi    = 120
		)
