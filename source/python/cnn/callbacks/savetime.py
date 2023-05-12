from torch.nn    import Module
from torch.optim import Optimizer

import torch

class SaveTimeModel (object) :

	def __init__ (self, filename : str, modulo : int = 250) -> None :
		"""
		Doc
		"""

		super().__init__()

		self.filename = filename
		self.modulo   = modulo

		if self.filename.count('[]') == 0 :
			tokens = self.filename.split('.')

			filename  = tokens[:-1]
			extension = tokens[-1]

			self.filename = '.'.join(filename) + '-[].' + extension

	def update (self, model : Module, optimizer : Optimizer, criterion : Module, epoch : int, loss : float) -> None :
		"""
		Doc
		"""

		if epoch > 0 and epoch % self.modulo == 0 :
			torch.save({
				'loss'      : loss,
				'epoch'     : epoch,
				'models'    : model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'criterion' : criterion
			}, self.filename.replace('[]', 'e{:04d}'.format(epoch)))
