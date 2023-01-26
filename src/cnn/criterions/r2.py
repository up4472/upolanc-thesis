from torch    import Tensor
from torch.nn import Module

from sklearn.metrics import r2_score

import numpy

class R2Score (Module) :

	def __init__ (self, reduction : str = 'mean') -> None :
		"""
		Doc
		"""

		super(R2Score, self).__init__()

		self.reduction = reduction

		match reduction.lower() :
			case 'none' : self.multioutput = 'raw_values'
			case 'mean' : self.multioutput = 'uniform_average'
			case 'sum'  : self.multioutput = 'raw_values'
			case _ : raise ValueError()

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
		if self.reduction != 'none' :
			score = numpy.expand_dims(score, axis = 0)

		score = Tensor(score)
		score = score.double()
		score = score.to(inputs.device)

		return score
