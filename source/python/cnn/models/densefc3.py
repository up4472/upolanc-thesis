from torch     import Tensor
from torch.nn  import Dropout
from torch.nn  import LeakyReLU
from torch.nn  import Linear
from torch.nn  import Module
from torchinfo import ModelStatistics
from torchinfo import torchinfo
from typing    import List

class DenseFC3 (Module) :

	def __init__ (self, input_size : int, output_size : int, hidden_size : List[int] = None, dropout : float = 0.1, leaky_relu : float = 0.0) -> None :
		"""
		Doc
		"""

		super(DenseFC3, self).__init__()

		if hidden_size is None :
			hidden_size = [256, 128]

		self.fc1 = Linear(in_features = input_size,     out_features = hidden_size[0])
		self.fc2 = Linear(in_features = hidden_size[0], out_features = hidden_size[1])
		self.fc3 = Linear(in_features = hidden_size[1], out_features = output_size)

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

		return 'N/A'

	@property
	def __str__ (self) -> str :
		"""
		Doc
		"""

		return 'N/A'

	def forward (self, x : Tensor) -> Tensor :
		"""
		Doc
		"""

		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc3(x)

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
	model = DenseFC3(
		input_size  = 64,
		output_size = 1,
		hidden_size = [256, 128],
		dropout     = 0.1,
		leaky_relu  = 0.0
	)

	model.summary(batch_size = 64, input_size = 64)
