from torch.nn                 import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data         import DataLoader
from typing                   import Any
from typing                   import Dict
from typing                   import List
from typing                   import Tuple

from ray import tune

import numpy
import os
import torch

from source.python.cnn.models import DenseFC2
from source.python.cnn.models import DenseFC3
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

def get_model_params (params : Dict[str, Any], config : Dict[str, Any], params_share : bool = False) -> Tuple[Dict, Dict] :
	"""
	Doc
	"""

	if not params_share :
		return params, config

	if 'model/convx/kernel' not in params.keys() :
		return params, config

	params['model/conv2/filters']  = params['model/convx/filters']
	params['model/conv2/kernel']   = params['model/convx/kernel']
	params['model/conv2/padding']  = params['model/convx/padding']
	params['model/conv2/dilation'] = params['model/convx/dilation']
	params['model/conv3/filters']  = params['model/convx/filters']
	params['model/conv3/kernel']   = params['model/convx/kernel']
	params['model/conv3/padding']  = params['model/convx/padding']
	params['model/conv3/dilation'] = params['model/convx/dilation']

	if config['model/type'].startswith('washburn2019') :
		params['model/conv4/filters']  = params['model/convx/filters']
		params['model/conv4/kernel']   = params['model/convx/kernel']
		params['model/conv4/padding']  = params['model/convx/padding']
		params['model/conv4/dilation'] = params['model/convx/dilation']
		params['model/conv5/filters']  = params['model/convx/filters']
		params['model/conv5/kernel']   = params['model/convx/kernel']
		params['model/conv5/padding']  = params['model/convx/padding']
		params['model/conv5/dilation'] = params['model/convx/dilation']
		params['model/conv6/filters']  = params['model/convx/filters']
		params['model/conv6/kernel']   = params['model/convx/kernel']
		params['model/conv6/padding']  = params['model/convx/padding']
		params['model/conv6/dilation'] = params['model/convx/dilation']

	params['model/maxpool1/kernel']  = params['model/maxpoolx/kernel']
	params['model/maxpool1/padding'] = params['model/maxpoolx/padding']
	params['model/maxpool2/kernel']  = params['model/maxpoolx/kernel']
	params['model/maxpool2/padding'] = params['model/maxpoolx/padding']
	params['model/maxpool3/kernel']  = params['model/maxpoolx/kernel']
	params['model/maxpool3/padding'] = params['model/maxpoolx/padding']

	params.pop('model/convx/filters',    None)
	params.pop('model/convx/kernel',     None)
	params.pop('model/convx/padding',    None)
	params.pop('model/convx/dilation',   None)
	params.pop('model/maxpoolx/kernel',  None)
	params.pop('model/maxpoolx/padding', None)

	return params, config

def get_model (params : Dict[str, Any], config : Dict[str, Any], params_share : bool = False) -> Module :
	"""
	Doc
	"""

	params, config = get_model_params(
		params       = params,
		config       = config,
		params_share = params_share
	)

	if config['model/type'] == 'zrimec2020r' :
		model = Zrimec2020r(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size'],
			'model/features'       : config['model/features']
		})

	elif config['model/type'] == 'zrimec2020c' :
		model = Zrimec2020c(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size'],
			'model/fc3/heads'      : config['model/output/heads'],
			'model/features'       : config['model/features']
		})

	elif config['model/type'] == 'washburn2019r' :
		model = Washburn2019r(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size'],
			'model/features'       : config['model/features']
		})

	elif config['model/type'] == 'washburn2019c' :
		model = Washburn2019c(params = params | {
			'model/input/channels' : config['model/input/channels'],
			'model/input/height'   : config['model/input/height'],
			'model/input/width'    : config['model/input/width'],
			'model/input/features' : config['model/input/features'],
			'model/fc3/features'   : config['model/output/size'],
			'model/fc3/heads'      : config['model/output/heads'],
			'model/features'       : config['model/features']
		})

	elif config['model/type'] == 'densefc2' :
		model = DenseFC2(
			input_size  = config['model/input/features'],
			output_size = config['model/output/size'],
			hidden_size = [
				params['model/fc1/features']
			],
			dropout    = params['model/dropout'],
			leaky_relu = params['model/leakyrelu']
		)

	elif config['model/type'] == 'densefc3' :
		model = DenseFC3(
			input_size  = config['model/input/features'],
			output_size = config['model/output/size'],
			hidden_size = [
				params['model/fc1/features'],
				params['model/fc2/features']
			],
			dropout    = params['model/dropout'],
			leaky_relu = params['model/leakyrelu']
		)

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

	if 'dataset/split/generator' in config.keys() :
		generator = config['dataset/split/generator']
	else :
		generator = generate_group_shuffle_split

	return to_dataloaders(
		dataset     = dataset,
		generator   = generator,
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

def get_metrics (config : Dict[str, Any], n_classes : int = 3) -> Dict[str, Module] :
	"""
	Doc
	"""

	metrics = dict()

	if config['model/mode'] == 'regression' :
		metrics = {
			'r2'    : get_criterion(reduction = 'mean', weights = None, query = 'r2', output_size = config['model/output/size']),
			'mae'   : get_criterion(reduction = 'mean', weights = None, query = 'mae'),
			'mape'  : get_criterion(reduction = 'mean', weights = None, query = 'mape'),
			'wmape' : get_criterion(reduction = 'mean', weights = None, query = 'wmape')
		}

	if config['model/mode'] == 'classification' :
		metrics = {
			'entropy'  : get_criterion(reduction = 'mean', weights = None, query = 'entropy'),
			'accuracy' : get_criterion(reduction = 'mean', weights = None, query = 'accuracy', n_classes = n_classes),
			'auroc'    : get_criterion(reduction = 'mean', weights = None, query = 'auroc',    n_classes = n_classes),
			'f1'       : get_criterion(reduction = 'mean', weights = None, query = 'f1',       n_classes = n_classes),
			'matthews' : get_criterion(reduction = 'mean', weights = None, query = 'matthews', n_classes = n_classes)
		}

	return {
		k : v.to(config['core/device'])
		for k, v in metrics.items()
	}

def tune_report (train_report : Dict[str, Any], valid_report : Dict[str, Any], lr : float, mode : str) -> None :
	"""
	Doc
	"""

	if mode == 'regression' :
		tune.report(
			valid_loss  = valid_report['loss'],
			valid_r2    = numpy.mean(valid_report['metric']['r2']),
			valid_mae   = numpy.mean(valid_report['metric']['mae']),
			valid_mape  = numpy.mean(valid_report['metric']['mape']),
			valid_wmape = numpy.mean(valid_report['metric']['wmape']),
			train_loss  = train_report['loss'],
			train_r2    = numpy.mean(train_report['metric']['r2']),
			train_mae   = numpy.mean(train_report['metric']['mae']),
			train_mape  = numpy.mean(train_report['metric']['mape']),
			train_wmape = numpy.mean(train_report['metric']['wmape']),
			lr          = lr
		)

	if mode == 'classification' :
		tune.report(
			valid_loss     = valid_report['loss'],
			valid_accuracy = numpy.mean(valid_report['metric']['accuracy']),
			valid_auroc    = numpy.mean(valid_report['metric']['auroc']),
			valid_f1       = numpy.mean(valid_report['metric']['f1']),
			valid_matthews = numpy.mean(valid_report['metric']['matthews']),
			train_loss     = train_report['loss'],
			train_accuracy = numpy.mean(train_report['metric']['accuracy']),
			train_auroc    = numpy.mean(train_report['metric']['auroc']),
			train_f1       = numpy.mean(train_report['metric']['f1']),
			train_matthews = numpy.mean(valid_report['metric']['matthews']),
			lr             = lr
		)

def main_loop (model_params : Dict[str, Any], config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	model = model_params['model']

	optimizer = model_params['optimizer']
	scheduler = model_params['scheduler']

	for epoch in range(config['model/epochs']) :
		current_lr = optimizer.param_groups[0]['lr']

		train_report = train_epoch(
			model  = model,
			params = model_params,
			desc   = ''
		)

		valid_report = evaluate_epoch(
			model      = model,
			params     = model_params,
			desc       = '',
			validation = True
		)

		if scheduler is not None :
			if isinstance(scheduler, ReduceLROnPlateau) :
				scheduler.step(valid_report['loss'])
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

		tune_report(
			train_report = train_report,
			valid_report = valid_report,
			lr           = current_lr,
			mode         = config['model/mode']
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
		params       = tune_config,
		config       = core_config,
		params_share = core_config['params/share']
	)

	model_trainers = get_model_trainers(
		model  = model,
		config = tune_config,
		epochs = core_config['model/epochs']
	)

	main_loop(
		config       = core_config,
		model_params = {
			'model'     : model,
			'criterion' : model_trainers['criterion'],
			'optimizer' : model_trainers['optimizer'],
			'scheduler' : model_trainers['scheduler'],
			'device'    : core_config['core/device'],
			'verbose'   : False,
			'metrics'   : get_metrics(
				config    = core_config,
				n_classes = core_config['model/output/size']
			),
			'train_dataloader' : dataloaders[0],
			'valid_dataloader' : dataloaders[1],
			'test_dataloader'  : dataloaders[2]
		}
	)
