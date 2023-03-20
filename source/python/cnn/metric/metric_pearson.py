from torch        import Tensor
from torch.nn     import Module
from torchmetrics import PearsonCorrCoef

import torch

class Metric_Pearson (Module) :

	def __init__ (self, reduction : str = 'mean', output_size : int = 1, **kwargs) -> None : # noqa : unused argument **kwargs
		"""
		Doc
		"""

		super(Metric_Pearson, self).__init__()

		self.reduction = reduction.lower()
		self.module    = PearsonCorrCoef(num_outputs = output_size)

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 2 and inputs.size(dim = 0) == 1 or inputs.size(dim = 1) == 1 :
			inputs = torch.flatten(inputs)
			labels = torch.flatten(labels)

		score = self.module(inputs, labels)

		if self.reduction == 'mean' :
			score = torch.mean(score)

		return score
