from torch                       import Tensor
from torch.nn                    import Module
from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix

import torch

class Metric_Confusion (Module) :

	def __init__ (self, reduction : str = 'mean', task : str = 'binary', n_classes : int = 1, **kwargs) -> None : # noqa : unused kwargs
		"""
		Doc
		"""

		super(Metric_Confusion, self).__init__()

		self.reduction = reduction.lower()
		self.task      = task.lower()

		if self.task == 'multiclass' :
			self.module = MulticlassConfusionMatrix(num_classes = n_classes)

		elif self.task == 'binary' :
			self.module = BinaryConfusionMatrix(threshold = 0.5)

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

		return self.module(inputs, labels)
