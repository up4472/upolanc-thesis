from torch    import Tensor
from torch.nn import Module

import torch

class Accuracy (Module) :

	def __init__ (self, reduction : str = 'mean') -> None :
		"""
		Doc
		"""

		super(Accuracy, self).__init__()

		self.reduction = reduction

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		index    = inputs.argmax(dim = 1)
		accuracy = (index == labels).double()

		if self.reduction == 'mean' :
			accuracy = torch.mean(accuracy, dim = None)

		return accuracy
