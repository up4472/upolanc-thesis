from torch                       import Tensor
from torch.nn                    import Module
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import MulticlassF1Score

import torch

class Metric_F1 (Module) :

	def __init__ (self, reduction : str = 'mean', task : str = 'binary', n_classes : int = 1, top_k : int = 1, **kwargs) -> None : # noqa : unused kwargs
		"""
		Doc
		"""

		super(Metric_F1, self).__init__()

		self.reduction = reduction.lower()
		self.task      = task.lower()

		if self.task == 'multiclass' :
			if   self.reduction == 'mean' : self.module = MulticlassF1Score(num_classes = n_classes, top_k = top_k, average = 'micro')
			elif self.reduction == 'none' : self.module = MulticlassF1Score(num_classes = n_classes, top_k = top_k, average = 'none')
			elif self.reduction == 'sum'  : self.module = MulticlassF1Score(num_classes = n_classes, top_k = top_k, average = 'none')
			else : ValueError()

		elif self.task == 'binary' :
			self.module = BinaryF1Score(task = 'binary', threshold = 0.5)

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
