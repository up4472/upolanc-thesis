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

	def __init__ (self, config) :
		"""
		Doc
		"""

		super(DNAPooler, self).__init__()

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
