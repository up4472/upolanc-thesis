import os

from torch                    import Tensor
from torch.nn                 import BCELoss
from torch.nn                 import Conv1d
from torch.nn                 import Conv2d
from torch.nn                 import CrossEntropyLoss
from torch.nn                 import HuberLoss
from torch.nn                 import L1Loss
from torch.nn                 import Linear
from torch.nn                 import MSELoss
from torch.nn                 import Module
from torch.nn                 import NLLLoss
from torch.nn                 import SmoothL1Loss
from torch.optim              import Adam
from torch.optim              import Optimizer
from torch.optim              import SGD
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from typing                   import Any
from typing                   import Dict
from typing                   import Union

from torch.nn.init import calculate_gain
from torch.nn.init import kaiming_normal_
from torch.nn.init import kaiming_uniform_
from torch.nn.init import ones_
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_

import numpy
import torch

from source.python.cnn.metric import Metric_Accuracy
from source.python.cnn.metric import Metric_AP
from source.python.cnn.metric import Metric_AUROC
from source.python.cnn.metric import Metric_Confusion
from source.python.cnn.metric import Metric_Corrected_MSE
from source.python.cnn.metric import Metric_Corrected_RMSE
from source.python.cnn.metric import Metric_KL
from source.python.cnn.metric import Metric_F1
from source.python.cnn.metric import Metric_Jaccardi
from source.python.cnn.metric import Metric_MAPE
from source.python.cnn.metric import Metric_Matthews
from source.python.cnn.metric import Metric_Pearson
from source.python.cnn.metric import Metric_R2
from source.python.cnn.metric import Metric_SMAPE
from source.python.cnn.metric import Metric_Spearman
from source.python.cnn.metric import Metric_WMAPE
from source.python.cnn.metric import Metric_Weighted
from source.python.cnn.models import Washburn2019c
from source.python.cnn.models import Washburn2019r
from source.python.cnn.models import Zrimec2020c
from source.python.cnn.models import Zrimec2020r

from source.python.cnn.cnn_trainer import evaluate
from source.python.cnn.cnn_trainer import train
from source.python.io.loader       import load_torch

def _get_optimizer (model : Module, config : Dict[str, Any]) -> Optimizer :
	"""
	Doc
	"""

	if config['optimizer/name'] == 'adam' :
		beta1 = 0.900
		beta2 = 0.999

		if   'optimizer/beta1'    in config.keys() : beta1 = config['optimizer/beta1']
		elif 'optimizer/momentum' in config.keys() : beta1 = config['optimizer/momentum']
		else : print('Using default beta1 : {}'.format(beta1))

		if   'optimizer/beta2'    in config.keys() : beta2 = config['optimizer/beta2']
		else : print('Using default beta2 : {}'.format(beta2))

		return get_optimizer(
			model        = model,
			query        = config['optimizer/name'],
			lr           = config['optimizer/lr'],
			weight_decay = config['optimizer/decay'],
			betas        = (beta1, beta2)
		)

	elif config['optimizer/name'] == 'sgd' :
		momentum = 0.900

		if   'optimizer/momentum' in config.keys() : momentum = config['optimizer/momentum']
		elif 'optimizer/beta1'    in config.keys() : momentum = config['optimizer/beta1']
		else : print('Using default momentum : {}'.format(momentum))

		return get_optimizer(
			model        = model,
			query        = config['optimizer/name'],
			lr           = config['optimizer/lr'],
			weight_decay = config['optimizer/decay'],
			momentum     = momentum,
		)

	raise ValueError()

def _get_scheduler (optimizer : Optimizer, config : Dict[str, Any], epochs : int) -> Any :
	"""
	Doc
	"""

	if config['scheduler/name'] == 'plateau' :
		return get_scheduler(
			optimizer = optimizer,
			query     = config['scheduler/name'],
			factor    = config['scheduler/plateau/factor'],
			patience  = config['scheduler/plateau/patience'],
			mode      = 'min',
			min_lr    = 1e-8
		)

	elif config['scheduler/name'] == 'linear':
		return get_scheduler(
			optimizer    = optimizer,
			query        = config['scheduler/name'],
			start_factor = 1.0,
			end_factor   = config['scheduler/linear/factor'],
			total_iters  = 500
		)

	elif config['scheduler/name'] == 'step' :
		return get_scheduler(
			optimizer = optimizer,
			query     = config['scheduler/name'],
			gamma     = config['scheduler/step/factor'],
			step_size = config['scheduler/step/patience'],
		)

	elif config['scheduler/name'] == 'constant' :
		return get_scheduler(
			optimizer   = optimizer,
			query       = config['scheduler/name'],
			factor      = 1.0,
			total_iters = epochs,
		)

	elif config['scheduler/name'] == 'exponential' :
		return get_scheduler(
			optimizer = optimizer,
			query     = config['scheduler/name'],
			gamma     = config['scheduler/exponential/factor']
		)

	elif config['scheduler/name'] == 'none' :
		return None

	raise ValueError()

def get_model_trainers (model : Module, config : Dict[str, Any], epochs : int) -> Dict[str, Any] :
	"""
	Doc
	"""

	if config['criterion/name'] == 'mse'     and isinstance(model, (Zrimec2020c, Washburn2019c)) : config['criterion/name'] = 'entropy'
	if config['criterion/name'] == 'entropy' and isinstance(model, (Zrimec2020r, Washburn2019r)) : config['criterion/name'] = 'mse'
	if config['criterion/name'] == 'bce'     and isinstance(model, (Zrimec2020r, Washburn2019r)) : config['criterion/name'] = 'mse'

	criterion_args = {
		'query'     : config['criterion/name'],
		'reduction' : config['criterion/reduction'],
		'weights'   : None
	}

	if config['criterion/name'].startswith('corrected-') :
		if 'criterion/threshold' in config.keys() :
			criterion_args['threshold'] = config['criterion/threshold']

	criterion = get_criterion(
		**criterion_args
	)

	optimizer = _get_optimizer(
		model  = model,
		config = config
	)

	scheduler = _get_scheduler(
		optimizer = optimizer,
		config    = config,
		epochs    = epochs
	)

	return {
		'criterion' : criterion,
		'optimizer' : optimizer,
		'scheduler' : scheduler
	}

def get_criterion (query : str, reduction : str = 'mean', weights : Union[numpy.ndarray, Tensor] = None, **kwargs) -> Metric_Weighted :
	"""
	Doc
	"""

	if isinstance(weights, numpy.ndarray) :
		weights = torch.tensor(weights)

	query = query.lower()

	if   query == 'accuracy'       : callable_criterion = Metric_Accuracy
	elif query == 'ap'             : callable_criterion = Metric_AP
	elif query == 'auroc'          : callable_criterion = Metric_AUROC
	elif query == 'bce'            : callable_criterion = BCELoss
	elif query == 'confusion'      : callable_criterion = Metric_Confusion
	elif query == 'corrected-mse'  : callable_criterion = Metric_Corrected_MSE
	elif query == 'corrected-rmse' : callable_criterion = Metric_Corrected_RMSE
	elif query == 'entropy'        : callable_criterion = CrossEntropyLoss
	elif query == 'f1'             : callable_criterion = Metric_F1
	elif query == 'huber'          : callable_criterion = HuberLoss
	elif query == 'jaccardi'       : callable_criterion = Metric_Jaccardi
	elif query == 'kl'             : callable_criterion = Metric_KL
	elif query == 'mae'            : callable_criterion = L1Loss
	elif query == 'mape'           : callable_criterion = Metric_MAPE
	elif query == 'matthews'       : callable_criterion = Metric_Matthews
	elif query == 'mse'            : callable_criterion = MSELoss
	elif query == 'nll'            : callable_criterion = NLLLoss
	elif query == 'pearson'        : callable_criterion = Metric_Pearson
	elif query == 'r2'             : callable_criterion = Metric_R2
	elif query == 'smae'           : callable_criterion = SmoothL1Loss
	elif query == 'smape'          : callable_criterion = Metric_SMAPE
	elif query == 'spearman'       : callable_criterion = Metric_Spearman
	elif query == 'wmape'          : callable_criterion = Metric_WMAPE
	else : raise ValueError()

	return Metric_Weighted(
		criterion = callable_criterion,
		reduction = reduction,
		weights   = weights,
		**kwargs
	)

def get_optimizer (query : str, model : Module, **kwargs) -> Optimizer :
	"""
	Doc
	"""

	query = query.lower()

	if   query == 'adam' : callable_optimizer = Adam
	elif query == 'sgd'  : callable_optimizer = SGD
	else : raise ValueError()

	return callable_optimizer(model.parameters(), **kwargs)

def get_scheduler (query : str, optimizer : Optimizer, **kwargs) -> Any :
	"""
	Doc
	"""

	query = query.lower()

	if   query == 'plateau'     : callable_scheduler = ReduceLROnPlateau
	elif query == 'exponential' : callable_scheduler = ExponentialLR
	elif query == 'step'        : callable_scheduler = StepLR
	elif query == 'linear'      : callable_scheduler = LinearLR
	elif query == 'constant'    : callable_scheduler = ConstantLR
	else : raise ValueError()

	return callable_scheduler(optimizer = optimizer, **kwargs)

def glorot_normal_weight (layer : Module, nonlinearity : str = 'relu') -> None :
	"""
	Doc
	"""

	if isinstance(layer, (Conv1d, Conv2d, Linear)) :
		xavier_normal_(layer.weight, gain = calculate_gain(nonlinearity))

def glorot_uniform_weight (layer : Module, nonlinearity : str = 'relu') -> None :
	"""
	Doc
	"""

	if isinstance(layer, (Conv1d, Conv2d, Linear)) :
		xavier_uniform_(layer.weight, gain = calculate_gain(nonlinearity))

def he_uniform_weight (layer : Module, nonlinearity : str = 'relu') -> None :
	"""
	Doc
	"""

	if isinstance(layer, (Conv1d, Conv2d, Linear)) :
		kaiming_uniform_(tensor = layer.weight, mode = 'fan_in', nonlinearity = nonlinearity)

def he_normal_weight (layer : Module, nonlinearity : str = 'relu') -> None :
	"""
	Doc
	"""

	if isinstance(layer, (Conv1d, Conv2d, Linear)) :
		kaiming_normal_(tensor = layer.weight, mode = 'fan_in', nonlinearity = nonlinearity)

def zero_bias (layer : Module) -> None :
	"""
	Doc
	"""

	if isinstance(layer, (Conv1d, Conv2d, Linear)) :
		if layer.bias is not None :
			zeros_(tensor = layer.bias)

def one_bias (layer : Module) -> None :
	"""
	Doc
	"""

	if isinstance(layer, (Conv1d, Conv2d, Linear)) :
		if layer.bias is not None :
			ones_(tensor = layer.bias)

def train_regressor (model : Module, params : Dict[str, Any]) -> Dict[str, Any] :
	"""
	Doc
	"""

	return train(model = model, params = params, regression = True)

def train_classifier (model : Module, params : Dict[str, Any]) -> Dict[str, Any] :
	"""
	Doc
	"""

	return train(model = model, params = params, regression = False)

def eval_regressor (model : Module, params : Dict[str, Any]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return evaluate(model = model, params = params, regression = True)

def eval_classifier (model : Module, params : Dict[str, Any]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return evaluate(model = model, params = params, regression = False)

def load_from_pretrained (filename : str, model : Module, strict : bool = True) -> Module :
	"""
	Doc
	"""

	if os.path.exists(filename) :
		checkpoint = load_torch(
			filename = filename
		)

		model.load_state_dict(
			state_dict = checkpoint['models'],
			strict     = strict
		)

		print('Sucessfully loaded pretrained model : {}'.format(filename))
		print()

	else :
		model = model.double()
		model = model.apply(he_uniform_weight)
		model = model.apply(zero_bias)

		print('Could not load pretrained model : {}'.format(filename))
		print()

	return model
