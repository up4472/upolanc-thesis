from torch        import Tensor
from torch.nn     import Module
from torchmetrics import AveragePrecision

import torch
class Metric_AP (Module) :

	def __init__ (self, reduction : str = 'mean', n_classes : int = 1, top_k : int = 1, **kwargs) -> None : # noqa : unused kwargs
		"""
		Doc
		"""

		super(Metric_AP, self).__init__()

		self.reduction = reduction.lower()

		if   self.reduction == 'mean' : self.module = AveragePrecision(task = 'multiclass', num_classes = n_classes, top_k = top_k, average = 'macro')
		elif self.reduction == 'none' : self.module = AveragePrecision(task = 'multiclass', num_classes = n_classes, top_k = top_k, average = 'none')
		elif self.reduction == 'sum'  : self.module = AveragePrecision(task = 'multiclass', num_classes = n_classes, top_k = top_k, average = 'none')
		else : ValueError()

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 3 :
			inputs = torch.softmax(inputs, dim = 1)
			labels = labels.int()
		else :
			raise NotImplementedError()

		score = self.module(inputs, labels)

		if self.reduction == 'sum' :
			score = torch.sum(score, dim = None)

		return score
