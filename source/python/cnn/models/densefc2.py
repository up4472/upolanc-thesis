from torch     import Tensor
from torch.nn  import Dropout
from torch.nn  import LeakyReLU
from torch.nn  import Linear
from torch.nn  import Module
from torchinfo import ModelStatistics
from typing    import List
from typing    import Union

from torchinfo import torchinfo

class DenseFC2 (Module) :

	def __init__ (self, input_size : int, output_size : int, hidden_size : Union[List, int] = 256, dropout : float = 0.1, leaky_relu : float = 0.0) -> None :
		"""
		Doc
		"""

		super(DenseFC2, self).__init__()

		if isinstance(hidden_size, list) :
			hidden_size = hidden_size[0]

		self.fc1 = Linear(in_features = input_size,  out_features = hidden_size)
		self.fc2 = Linear(in_features = hidden_size, out_features = output_size)

		self.dropout = Dropout(
			p       = dropout,
			inplace = False
		)

		self.relu = LeakyReLU(
			negative_slope = leaky_relu,
			inplace        = False
		)

	@property
	def __name__ (self) -> str :
		"""
		Doc
		"""

		return 'densefc2'

	@property
	def __str__ (self) -> str :
		"""
		Doc
		"""

		return 'N/A'

	def forward (self, x : Tensor, v : Tensor = None) -> Tensor :
		"""
		Doc
		"""

		if v is not None :
			x = v

		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc2(x)

		return x

	def summary (self, batch_size : int, input_size : int, verbose : bool = False) -> ModelStatistics :
		""""
		Doc
		"""

		col_names = ['input_size', 'output_size', 'num_params']

		if verbose :
			col_names.extend(['kernel_size', 'mult_adds', 'trainable'])

		return torchinfo.summary(self,
			input_size = (batch_size, input_size),
			col_names  = col_names
		)

if __name__ == '__main__' :
	model = DenseFC2(
		input_size  = 64,
		output_size = 1,
		hidden_size = 256,
		dropout     = 0.1,
		leaky_relu  = 0.0
	)

	model.summary(batch_size = 64, input_size = 64)
