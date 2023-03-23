from torch        import Tensor
from torch.nn     import Module
from torchmetrics import MeanAbsolutePercentageError

import torch

class Metric_MAPE (Module) :

	def __init__ (self, reduction : str = 'mean', **kwargs) -> None : # noqa : unused argument **kwargs
		"""
		Doc
		"""

		super(Metric_MAPE, self).__init__()

		self.module = MeanAbsolutePercentageError()

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 2 and inputs.size(dim = 0) == 1 or inputs.size(dim = 1) == 1 :
			inputs = torch.flatten(inputs)
			labels = torch.flatten(labels)

		return self.module(inputs, labels)
