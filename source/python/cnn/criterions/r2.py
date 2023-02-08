from torch    import Tensor
from torch.nn import Module

from sklearn.metrics import r2_score

import numpy
import torch

class R2Score (Module) :

	def __init__ (self, reduction : str = 'mean', **kwargs) -> None : # noqa : unused kwargs
		"""
		Doc
		"""

		super(R2Score, self).__init__()

		self.reduction = reduction.lower()

		if   self.reduction == 'none' : self.multioutput = 'raw_values'
		elif self.reduction == 'mean' : self.multioutput = 'uniform_average'
		elif self.reduction == 'sum'  : self.multioutput = 'raw_values'
		else : raise ValueError()

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		numpy_labels = labels.detach().cpu().numpy()
		numpy_inputs = inputs.detach().cpu().numpy()

		score = r2_score(
			y_true = numpy_labels,
			y_pred = numpy_inputs,
			multioutput = self.multioutput
		)

		if self.reduction == 'sum' :
			score = numpy.sum(score)

		score = torch.tensor(score)
		score = score.double()
		score = score.to(inputs.device)

		return score
