from torch    import Tensor
from torch.nn import Module

from sklearn.metrics import r2_score

class R2Score (Module) :

	def __init__ (self) -> None :
		"""
		Doc
		"""

		super(R2Score, self).__init__()

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor : # noqa : make static
		"""
		Doc
		"""

		numpy_labels = labels.detach().cpu().numpy()
		numpy_inputs = inputs.detach().cpu().numpy()

		score = r2_score(
			y_true = numpy_labels,
			y_pred = numpy_inputs,
			multioutput = 'raw_values'
		)

		score = Tensor(score)
		score = score.double()
		score = score.to(inputs.device)

		return score
