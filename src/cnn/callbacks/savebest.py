from torch.nn    import Module
from torch.optim import Optimizer

import numpy
import torch

class SaveBestModel (object) :

	def __init__ (self, filename : str, loss : float = numpy.inf) -> None :
		"""
		Doc
		"""

		super().__init__()

		self.filename = filename
		self.loss = loss

	def update (self, model : Module, optimizer : Optimizer, criterion : Module, epoch : int, loss : float) -> None :
		"""
		Doc
		"""

		if self.loss > loss :
			self.loss = loss

			torch.save({
				'loss'      : loss,
				'epoch'     : epoch,
				'models'    : model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'criterion' : criterion
			}, self.filename)
