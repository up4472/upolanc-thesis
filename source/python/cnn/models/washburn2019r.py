from torch     import Tensor
from torch.nn  import Linear
from torch.nn  import Module
from torch.nn  import ReLU
from torchinfo import ModelStatistics
from typing    import Any
from typing    import Dict

import torchinfo

from source.python.cnn.models.washburn2019 import Washburn2019
from source.python.cnn.models.washburn2019 import update_params

class Washburn2019r (Module) :

	def __init__ (self, params : Dict[str, Any] = None) -> None :
		"""
		Doc
		"""

		super(Washburn2019r, self).__init__()

		params = update_params(params = params)

		self.backbone = Washburn2019(
			params = params
		)

		self.fc3 = Linear(
			in_features  = params['fc2']['features'],
			out_features = params['fc3']['features']
		)

		self.relu = ReLU(inplace = False)

	@property
	def __name__ (self) -> str :
		"""
		Doc
		"""

		return 'washburn2019r'

	@property
	def __str__ (self) -> str :
		"""
		Doc
		"""

		return 'Washburn et. al (2019)'

	def forward (self, x : Tensor, v : Tensor = None) -> Tensor :
		"""
		Doc
		"""

		x = self.backbone(x, v)
		x = self.fc3(x)
		x = self.relu(x)

		return x

	def summary (self, batch_size : int, in_channels : int, in_height : int, in_width : int, in_features : int, verbose : bool = False) -> ModelStatistics :
		""""
		Doc
		"""

		col_names = ['input_size', 'output_size', 'num_params']

		if verbose :
			col_names.extend(['kernel_size', 'mult_adds', 'trainable'])

		return torchinfo.summary(self,
			input_size = [(batch_size, in_channels, in_height, in_width), (batch_size, in_features)],
			col_names  = col_names
		)

if __name__ == '__main__' :
	model = Washburn2019r(params = {
		'other' : {
			'in_channels' : 1,
			'in_height'   : 4,
			'in_width'    : 2150,
			'in_features' : 64
		},
		'fc3' : {
			'features' : 8
		}
	})

	model.summary(batch_size = 64, in_channels = 1, in_height = 4, in_width = 2150, in_features = 64)
