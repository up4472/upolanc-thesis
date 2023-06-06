from torch        import Tensor
from torch.nn     import Module
from torchmetrics import MeanSquaredError
from torchmetrics import MeanAbsolutePercentageError

import torch

class Metric_Corrected_MSE (Module) :

	def __init__ (self, reduction : str = 'mean', threshold : float = 0.0, **kwargs) -> None : # noqa : unused argument **kwargs
		"""
		Doc
		"""

		super(Metric_Corrected_MSE, self).__init__()

		if reduction.lower() != 'mean' :
			raise ValueError()

		self.module_se = MeanSquaredError(squared = True)
		self.module_pe = MeanAbsolutePercentageError()

		self.threshold = threshold

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 2 and inputs.size(dim = 0) == 1 or inputs.size(dim = 1) == 1 :
			inputs = torch.flatten(inputs)
			labels = torch.flatten(labels)

		score_se = self.module_se(inputs, labels)
		score_pe = self.module_pe(inputs, labels)

		score = torch.where(score_pe > self.threshold, score_se, 0.0)

		return score
