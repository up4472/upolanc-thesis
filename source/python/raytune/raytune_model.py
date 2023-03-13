from torch.nn                 import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data         import DataLoader
from typing                   import Any
from typing                   import Dict
from typing                   import List

from ray import tune

import numpy
import os
import torch

from source.python.cnn.models  import Washburn2019c
from source.python.cnn.models  import Washburn2019r
from source.python.cnn.models  import Zrimec2020c
from source.python.cnn.models  import Zrimec2020r

from source.python.cnn.cnn_trainer         import evaluate_epoch
from source.python.cnn.cnn_trainer         import train_epoch
from source.python.dataset.dataset_classes import GeneDataset
from source.python.runtime                 import lock_random
from source.python.dataset.dataset_split   import generate_group_shuffle_split
from source.python.dataset.dataset_utils   import to_dataloaders
from source.python.cnn.cnn_model           import get_criterion
from source.python.cnn.cnn_model           import get_model_trainers
from source.python.cnn.cnn_model           import he_uniform_weight
from source.python.cnn.cnn_model           import zero_bias

def get_model (params : Dict[str, Any], config : Dict[str, Any]) -> Module :
	"""
	Doc
	"""

	if config['model/type'] == 'zrimec2020r' :
		model = Zrimec2020r(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size']
		})

	elif config['model/type'] == 'zrimec2020c' :
		model = Zrimec2020c(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size'],
			'model/fc3/heads'      : config['model/output/heads']
		})

	elif config['model/type'] == 'washburn2019r' :
		model = Washburn2019r(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size']
		})

	elif config['model/type'] == 'washburn2019c' :
		model = Washburn2019c(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size'],
			'model/fc3/heads'      : config['model/output/heads']
		})

	else :
		raise ValueError()

	model = model.double()
	model = model.apply(he_uniform_weight)
	model = model.apply(zero_bias)
	model = model.to(config['core/device'])

	return model

def get_dataloaders (params : Dict[str, Any], config : Dict[str, Any], dataset : GeneDataset = None) -> List[DataLoader] :
	"""
	Doc
	"""

	if dataset is None :
		dataset = torch.load(config['dataset/filepath'])

	return to_dataloaders(
		dataset     = dataset,
		generator   = generate_group_shuffle_split,
		random_seed = config['core/random'],
		split_size  = {
			'valid' : config['dataset/split/valid'],
			'test'  : config['dataset/split/test']
		},
		batch_size  = {
			'train' : params['dataset/batch_size'],
			'valid' : params['dataset/batch_size'],
			'test'  : params['dataset/batch_size']
		}
	)

def get_metrics (config : Dict[str, Any]) -> Dict[str, Module] :
	"""
	Doc
	"""

	if config['model/type'].endswith('r') :
		metrics = {
			'r2'    : get_criterion(reduction = 'mean', weights = None, query = 'r2', output_size = config['model/output/size']),
			'mae'   : get_criterion(reduction = 'mean', weights = None, query = 'mae'),
			'mape'  : get_criterion(reduction = 'mean', weights = None, query = 'mape'),
			'wmape' : get_criterion(reduction = 'mean', weights = None, query = 'wmape')
		}
	else :
		metrics = {
			'entropy'  : get_criterion(reduction = 'mean', weights = None, query = 'entropy'),
			'accuracy' : get_criterion(reduction = 'mean', weights = None, query = 'accuracy')
		}

	return {
		k : v.to(config['core/device'])
		for k, v in metrics.items()
	}

def regression_loop (model_params : Dict[str, Any], config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	model = model_params['model']

	optimizer = model_params['optimizer']
	scheduler = model_params['scheduler']

	for epoch in range(config['model/epochs']) :
		current_lr = optimizer.param_groups[0]['lr']

		train_report = train_epoch(model = model, params = model_params, desc = '')
		valid_report = evaluate_epoch(model = model, params = model_params, desc = '', validation = True)

		train_loss  = train_report['loss']
		train_r2    = train_report['metric']['r2']
		train_mae   = train_report['metric']['mae']
		train_mape  = train_report['metric']['mape']
		train_wmape = train_report['metric']['wmape']

		valid_loss  = valid_report['loss']
		valid_r2    = valid_report['metric']['r2']
		valid_mae   = valid_report['metric']['mae']
		valid_mape  = valid_report['metric']['mape']
		valid_wmape = valid_report['metric']['wmape']

		if scheduler is not None :
			if isinstance(scheduler, ReduceLROnPlateau) :
				scheduler.step(valid_loss)
			else :
				scheduler.step()

		if config['tuner/checkpoint'] :
			with tune.checkpoint_dir(epoch) as checkpoint :
				path = os.path.join(checkpoint, 'checkpoint')
				data = (
					model.state_dict(),
					optimizer.state_dict()
				)

				torch.save(data, path)

		tune.report(
			valid_loss  = valid_loss,
			valid_r2    = numpy.mean(valid_r2),
			valid_mae   = numpy.mean(valid_mae),
			valid_mape  = numpy.mean(valid_mape),
			valid_wmape = numpy.mean(valid_wmape),
			train_loss  = train_loss,
			train_r2    = numpy.mean(train_r2),
			train_mae   = numpy.mean(train_mae),
			train_mape  = numpy.mean(train_mape),
			train_wmape = numpy.mean(train_wmape),
			lr          = current_lr
		)

def main (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	lock_random(seed = core_config['core/random'])

	dataloaders = get_dataloaders(
		params  = tune_config,
		config  = core_config,
		dataset = None
	)

	model = get_model(
		params = tune_config,
		config = core_config
	)

	model_trainers = get_model_trainers(
		model  = model,
		config = tune_config,
		epochs = core_config['model/epochs']
	)

	regression_loop(
		config       = core_config,
		model_params = {
			'model'     : model,
			'criterion' : model_trainers['criterion'],
			'optimizer' : model_trainers['optimizer'],
			'scheduler' : model_trainers['scheduler'],
			'device'    : core_config['core/device'],
			'verbose'   : False,
			'metrics'   : get_metrics(config = core_config),
			'train_dataloader' : dataloaders[0],
			'valid_dataloader' : dataloaders[1],
			'test_dataloader'  : dataloaders[2]
		}
	)
