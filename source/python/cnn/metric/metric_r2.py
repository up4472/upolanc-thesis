from torch        import Tensor
from torch.nn     import Module
from torchmetrics import R2Score

import torch

class Metric_R2 (Module) :

	def __init__ (self, reduction : str = 'mean', output_size : int = 1, force_finite : bool = False, **kwargs) -> None : # noqa : unused argument **kwargs
		"""
		Doc
		"""

		super(Metric_R2, self).__init__()

		self.reduction    = reduction.lower()
		self.multioutput  = None
		self.force_finite = force_finite

		if   self.reduction == 'none' : self.multioutput = 'raw_values'
		elif self.reduction == 'mean' : self.multioutput = 'uniform_average'
		else : raise ValueError()

		self.module = R2Score(
			num_outputs = output_size,
			adjusted    = 0,
			multioutput = self.multioutput
		)

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		if inputs.dim() == 2 and inputs.size(dim = 0) == 1 or inputs.size(dim = 1) == 1 :
			inputs = torch.flatten(inputs)
			labels = torch.flatten(labels)

		score = self.module(inputs, labels)

		if self.force_finite :
			score = torch.nan_to_num(
				input  = score,
				nan    = 1.0,
				posinf = 0.0,
				neginf = 0.0
			)

		return score
