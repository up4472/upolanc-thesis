from torch    import Tensor
from torch.nn import Module

from sklearn.metrics import brier_score_loss

class BrierScore (Module) :

	def __init__ (self) -> None :
		"""
		Doc
		"""

		super(BrierScore, self).__init__()

		self.weights = None

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		numpy_labels = labels.detach().cpu().numpy()
		numpy_inputs = inputs.detach().cpu().numpy()

		score = brier_score_loss(
			y_true = numpy_labels,
			y_prob = numpy_inputs,
			sample_weight = self.weights
		)

		score = Tensor(score)
		score = score.double()
		score = score.to(inputs.device)

		return score
