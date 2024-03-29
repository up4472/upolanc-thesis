from torch                       import Tensor
from torch.nn                    import Module
from torchmetrics.classification import BinaryMatthewsCorrCoef
from torchmetrics.classification import MulticlassMatthewsCorrCoef

import torch

class Metric_Matthews (Module) :

	def __init__ (self, reduction : str = 'mean', task : str = 'binary', n_classes : int = 1, top_k : int = 1, **kwargs) -> None : # noqa : unused kwargs
		"""
		Doc
		"""

		super(Metric_Matthews, self).__init__()

		self.reduction = reduction.lower()
		self.task      = task.lower()

		if self.task == 'multiclass' :
			self.module = MulticlassMatthewsCorrCoef(num_classes = n_classes)

		elif self.task == 'binary' :
			self.module = BinaryMatthewsCorrCoef(threshold = 0.5)

		else :
			raise ValueError()

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 3 :
			inputs = torch.softmax(inputs, dim = 1)
			labels = labels.int()
		else :
			labels = labels.int()

		score = self.module(inputs, labels)

		if self.reduction == 'sum' :
			score = torch.sum(score, dim = None)

		return score
