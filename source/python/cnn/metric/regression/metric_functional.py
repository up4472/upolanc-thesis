from torch         import Tensor
from torchmetrics import WeightedMeanAbsolutePercentageError

import torch

def compute_wmape (inputs : Tensor, labels : Tensor, eps : float = 1e-7) -> Tensor : # noqa : unused parameter
	"""
	Doc
	"""

	module = WeightedMeanAbsolutePercentageError()
	module = module.to(inputs.device)

	return module(inputs, labels)

def compute_smape (inputs : Tensor, labels : Tensor, eps : float = 1e-7) -> Tensor :
	"""
	Doc
	"""

	a = labels
	f = inputs
	d = eps

	x = torch.sub(input = f, other = a)
	x = torch.abs(input = x)

	y = torch.add(
		input = torch.abs(input = f),
		other = torch.abs(input = a)
	)

	y = torch.add(input = y, other = d)
	y = torch.div(input = y, other = 2)
	p = torch.div(input = x, other = y)

	return p

def compute_mape (inputs : Tensor, labels : Tensor, eps : float = 1e-7) -> Tensor :
	"""
	Doc
	"""

	a = labels
	f = inputs
	d = eps

	x = torch.sub(input = a, other = f)
	y = torch.add(input = a, other = d)

	p = torch.div(input = x, other = y)
	p = torch.abs(input = p)

	return p
