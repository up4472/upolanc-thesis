import torch
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Tanh

class DefaultPooler (Module) :

	def __init__ (self, config) :
		"""
		Doc
		"""

		super(DefaultPooler, self).__init__()

		self.fc1  = Linear(config.hidden_size, config.hidden_size)
		self.tanh = Tanh()

	def forward (self, hidden_states) :
		"""
		Doc
		"""

		token = hidden_states[:, 0]

		output = self.fc1(token)
		output = self.tanh(output)

		return output

class DNAPooler (Module) :

	def __init__ (self, config) : # noqa unused
		"""
		Doc
		"""

		super(DNAPooler, self).__init__()

		self.lindex = None
		self.rindex = None

		self.custom_range = False

		if self.lindex is not None : self.custom_range = True
		if self.rindex is not None : self.custom_range = True

	def forward (self, hidden_states) :
		"""
		Doc
		"""

		if self.custom_range :
			hidden_states = hidden_states[self.lindex : self.rindex, :]

		return torch.mean(hidden_states, dim = 1)
