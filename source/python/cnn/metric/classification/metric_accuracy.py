from torch                       import Tensor
from torch.nn                    import Module
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassAccuracy

import torch

class Metric_Accuracy (Module) :

	def __init__ (self, reduction : str = 'mean', task : str = 'binary', n_classes : int = 1, top_k : int = 1, **kwargs) -> None : # noqa : unused kwargs
		"""
		Doc
		"""

		super(Metric_Accuracy, self).__init__()

		self.reduction = reduction.lower()
		self.task      = task.lower()

		if self.task == 'multiclass' :
			if   self.reduction == 'mean' : self.module = MulticlassAccuracy(num_classes = n_classes, top_k = top_k, average = 'micro')
			elif self.reduction == 'none' : self.module = MulticlassAccuracy(num_classes = n_classes, top_k = top_k, average = 'none')
			elif self.reduction == 'sum'  : self.module = MulticlassAccuracy(num_classes = n_classes, top_k = top_k, average = 'none')
			else : ValueError()

		elif self.task == 'binary' :
			self.module = BinaryAccuracy(threshold = 0.5)

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
