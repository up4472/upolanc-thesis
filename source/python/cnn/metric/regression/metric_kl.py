from torch        import Tensor
from torch.nn     import Module
from torchmetrics import KLDivergence

import torch

class Metric_KL (Module) :

	def __init__ (self, reduction : str = 'mean', log_prob : bool = False, **kwargs) -> None : # noqa : unused argument **kwargs
		"""
		Doc
		"""

		super(Metric_KL, self).__init__()

		self.reduction = reduction.lower()

		if   self.reduction == 'mean' : self.module = KLDivergence(log_prob = log_prob, reduction = 'mean')
		elif self.reduction == 'none' : self.module = KLDivergence(log_prob = log_prob, reduction = 'none')
		elif self.reduction == 'sum'  : self.module = KLDivergence(log_prob = log_prob, reduction = 'sum')
		else : raise ValueError()

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 2 and inputs.size(dim = 0) == 1 or inputs.size(dim = 1) == 1 :
			inputs = torch.flatten(inputs)
			labels = torch.flatten(labels)

		return self.module(inputs, labels)
