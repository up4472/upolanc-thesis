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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from typing                   import Any
from typing                   import Dict
from typing                   import Tuple
from typing                   import Union

from torch.nn.init import calculate_gain
from torch.nn.init import kaiming_normal_
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_

import numpy

from src.cnn.criterions import Accuracy
from src.cnn.criterions import BrierScore
from src.cnn.criterions import R2Score

from src.cnn._common import evaluate
from src.cnn._common import train

def get_criterion (query : str, reduction : str = 'mean', return_name : bool = False, **kwargs) -> Union[Module, Tuple[Module, str]] :
	"""
	Doc
	"""

	match query.lower() :
		case 'mse' :
			func = MSELoss(reduction = reduction, **kwargs)
			name = 'Mean Squared Error'
		case 'mae' :
			func = L1Loss(reduction = reduction, **kwargs)
			name = 'Mean Absolute Error'
		case 'smooth-mae' :
			func = SmoothL1Loss(reduction = reduction, **kwargs)
			name = 'Smooth Mean Absolute Error'
		case 'huber' :
			func = HuberLoss(reduction = reduction, **kwargs)
			name = 'Huber'
		case 'r2' :
			func = R2Score()
			name = 'R2 Score'
		case 'entropy' :
			func = CrossEntropyLoss(reduction = reduction, **kwargs)
			name = 'Cross Entropy'
		case 'brier' :
			func = BrierScore()
			name = 'Brier'
		case 'nll' :
			func = NLLLoss(reduction = reduction, **kwargs)
			name = 'Negative Log Likelihood'
		case 'acc' :
			func = Accuracy(reduction = reduction)
			name = 'Accuracy'
		case _ :
			raise ValueError()

	if return_name :
		return func, name

	return func

def get_optimizer (query : str, model : Module, return_name : bool = False, **kwargs) -> Union[Optimizer, Tuple[Optimizer, str]] :
	"""
	Doc
	"""

	match query.lower() :
		case 'adam' :
			func = Adam(model.parameters(), **kwargs)
			name = 'ADAM'
		case 'sgd'  :
			func = SGD(model.parameters(), **kwargs)
			name = 'SGD'
		case _ :
			raise ValueError()

	if return_name :
		return func, name

	return func

def get_scheduler (query : str, optimizer : Optimizer, return_name : bool = False, **kwargs) -> Union[Any, Tuple[Any, str]] :
	"""
	Doc
	"""

	match query.lower() :
		case 'plateau' :
			func = ReduceLROnPlateau(optimizer = optimizer, **kwargs)
			name = 'Reduce On Plateau'
		case 'exponential' :
			func = ExponentialLR(optimizer = optimizer, **kwargs)
			name = 'Exponential'
		case 'step' :
			func = StepLR(optimizer = optimizer, **kwargs)
			name = 'Step'
		case 'conststant' :
			func = ConstantLR(optimizer = optimizer, **kwargs)
			name = 'Constant'
		case _ :
			func = None
			name = 'None'

	if return_name :
		return func, name

	return func

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

def train_regressor (model : Module, params : Dict[str, Any]) -> Dict[str, Dict | numpy.ndarray] :
	"""
	Doc
	"""

	return train(model = model, params = params)

def train_classifier (model : Module, params : Dict[str, Any]) -> Dict[str, Dict | numpy.ndarray] :
	"""
	Doc
	"""

	return train(model = model, params = params)

def eval_regressor (model: Module, params: Dict[str, Any]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return evaluate(model = model, params = params)

def eval_classifier (model: Module, params: Dict[str, Any]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return evaluate(model = model, params = params)
