from torch    import Tensor
from torch.nn import Module
from torch.nn import MSELoss

import torch

from source.python.cnn.metric.regression.metric_functional import compute_mape

class Metric_Corrected_RMSE (Module) :

	def __init__ (self, reduction : str = 'mean', threshold : float = 0.0, **kwargs) -> None : # noqa : unused argument **kwargs
		"""
		Doc
		"""

		super(Metric_Corrected_RMSE, self).__init__()

		self.reduction = reduction.lower()
		self.threshold = threshold
		self.eps       = 1e-7

		self.module = MSELoss(reduction = 'none')

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 2 and inputs.size(dim = 0) == 1 or inputs.size(dim = 1) == 1 :
			inputs = torch.flatten(inputs)
			labels = torch.flatten(labels)

		x = self.module(inputs, labels)

		y = compute_mape(
			inputs = inputs,
			labels = labels,
			eps    = self.eps
		)

		x = torch.sqrt(input = x)

		score = torch.where(y > self.threshold, x, 0.0)

		if   self.reduction == 'mean' : return torch.mean(score)
		elif self.reduction == 'sum'  : return torch.sum(score)
		elif self.reduction == 'none' : return score

		raise ValueError()
