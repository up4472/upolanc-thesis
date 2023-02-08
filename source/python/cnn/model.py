from torch                    import Tensor
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

from source.python.cnn.criterions import Accuracy
from source.python.cnn.criterions import R2Score
from source.python.cnn.criterions import WeightedCriterion

from source.python.cnn._common import evaluate
from source.python.cnn._common import train

def get_criterion (query : str, reduction : str = 'mean', weights : Union[numpy.ndarray, Tensor] = None, **kwargs) -> WeightedCriterion :
	"""
	Doc
	"""

	if isinstance(weights, numpy.ndarray) :
		weights = torch.tensor(weights)

	query = query.lower()

	if   query == 'mse'        : callable_criterion = MSELoss
	elif query == 'mae'        : callable_criterion = L1Loss
	elif query == 'smooth-mae' : callable_criterion = SmoothL1Loss
	elif query == 'huber'      : callable_criterion = HuberLoss
	elif query == 'r2'         : callable_criterion = R2Score
	elif query == 'entropy'    : callable_criterion = CrossEntropyLoss
	elif query == 'nll'        : callable_criterion = NLLLoss
	elif query == 'accuracy'   : callable_criterion = Accuracy
	else : raise ValueError()

	return WeightedCriterion(
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

def train_regressor (model : Module, params : Dict[str, Any]) -> Dict[str, Dict | numpy.ndarray] :
	"""
	Doc
	"""

	return train(model = model, params = params, regression = True)

def train_classifier (model : Module, params : Dict[str, Any]) -> Dict[str, Dict | numpy.ndarray] :
	"""
	Doc
	"""

	return train(model = model, params = params, regression = False)

def eval_regressor (model: Module, params: Dict[str, Any]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return evaluate(model = model, params = params, regression = True)

def eval_classifier (model: Module, params: Dict[str, Any]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return evaluate(model = model, params = params, regression = False)
