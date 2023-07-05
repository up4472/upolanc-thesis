from torch    import Tensor
from torch.nn import Module
from typing   import Callable

import torch

from source.python.cnn.metric.regression.metric_r2 import Metric_R2

class Metric_Weighted (Module) :

	def __init__ (self, criterion : Callable, reduction : str = 'mean', weights : Tensor = None, **kwargs) -> None :
		"""
		Doc
		"""

		super(Metric_Weighted, self).__init__()

		self.reduction = reduction.lower()
		self.weights   = weights

		if weights is None : self.criterion = criterion(reduction = self.reduction, **kwargs)
		else               : self.criterion = criterion(reduction = 'none',         **kwargs)

		self.flag1 = self.weights is not None
		self.flag2 = self.reduction != 'none'

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		score = self.criterion(inputs, labels)

		if self.flag1 and self.flag2 :
			if not isinstance(self.criterion, Metric_R2) :
				score = torch.mean(score, dim = 0)

			score = torch.dot(self.weights, score)

		return score
