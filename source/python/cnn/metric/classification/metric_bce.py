from torch    import Tensor
from torch.nn import Module
from typing   import Dict

import torch

class Metric_BCE (Module) :

	def __init__ (self, weights : Dict[int, float] = None, eps : float = 1e-7, **kwargs) -> None : # noqa : unused kwargs
		"""
		Doc
		"""

		super(Metric_BCE, self).__init__()

		if weights is None :
			weights = {
				0 : 1,
				1 : 1
			}

		if 0 not in weights.keys() : raise KeyError()
		if 1 not in weights.keys() : raise KeyError()

		self.weights = weights
		self.eps     = eps

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() != 2 :
			raise ValueError()

		inputs = inputs.flatten()
		labels = labels.flatten().int()

		plabels =       labels
		nlabels = 1.0 - labels
		pinputs =       inputs
		ninputs = 1.0 - inputs

		score_1 = self.weights[1] * (plabels * torch.log(pinputs + self.eps))
		score_0 = self.weights[0] * (nlabels * torch.log(ninputs + self.eps))

		return torch.neg(torch.mean(score_1 + score_0))
