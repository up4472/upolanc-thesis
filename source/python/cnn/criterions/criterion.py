from torch    import Tensor
from torch.nn import Module
from typing   import Callable

import torch

from source.python.cnn.criterions import R2Score

class WeightedCriterion (Module) :

	def __init__ (self, criterion : Callable, reduction : str = 'mean', weights : Tensor = None, **kwargs) -> None :
		"""
		Doc
		"""

		super(WeightedCriterion, self).__init__()

		self.vectorized = False
		self.reduction  = reduction.lower()
		self.weights    = weights

		if self.reduction not in ['none', 'mean', 'sum'] :
			raise ValueError()

		if weights is not None :
			self.criterion = criterion(reduction = 'none', **kwargs)
		else :
			self.criterion = criterion(reduction = self.reduction, **kwargs)

		if isinstance(self.criterion, R2Score) :
			self.vectorized = True

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		score = self.criterion(inputs, labels)

		if self.weights is not None and self.reduction != 'none' :
			if not self.vectorized :
				score = torch.mean(score, dim = 0)

			score = torch.dot(self.weights, score)

		return score
