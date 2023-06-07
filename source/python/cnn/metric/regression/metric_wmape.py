from torch    import Tensor
from torch.nn import Module

import torch

from source.python.cnn.metric.regression.metric_functional import compute_wmape

class Metric_WMAPE (Module) :

	def __init__ (self, reduction : str = 'mean', **kwargs) -> None : # noqa : unused argument **kwargs
		"""
		Doc
		"""

		super(Metric_WMAPE, self).__init__()

		self.reduction = reduction.lower()
		self.eps       = 1e-7

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 2 and inputs.size(dim = 0) == 1 or inputs.size(dim = 1) == 1 :
			inputs = torch.flatten(inputs)
			labels = torch.flatten(labels)

		p = compute_wmape(
			inputs = inputs,
			labels = labels,
			eps    = self.eps
		)

		if   self.reduction == 'mean' : return torch.mean(p)
		elif self.reduction == 'sum'  : return torch.sum(p)
		elif self.reduction == 'none' : return p

		raise ValueError()

