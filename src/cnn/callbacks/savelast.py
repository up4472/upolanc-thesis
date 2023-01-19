from torch.nn    import Module
from torch.optim import Optimizer

import torch

class SaveLastModel (object) :

	def __init__ (self, filename : str, epoch : int = -1) -> None :
		"""
		Doc
		"""

		super().__init__()

		self.filename = filename
		self.epoch = epoch

	def update (self, model : Module, optimizer : Optimizer, criterion : Module, epoch : int, loss : float) -> None :
		"""
		Doc
		"""

		if self.epoch < epoch :
			self.epoch = epoch

			torch.save({
				'loss'      : loss,
				'epoch'     : epoch,
				'models'    : model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'criterion' : criterion
			}, self.filename)
