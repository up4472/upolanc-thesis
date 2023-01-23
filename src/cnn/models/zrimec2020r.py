from torch     import Tensor
from torch.nn  import Linear
from torch.nn  import Module
from torch.nn  import ReLU
from torchinfo import ModelStatistics
from typing    import Dict

import torchinfo

from src.cnn.models.zrimec2020 import Zrimec2020
from src.cnn.models.zrimec2020 import update_params

class Zrimec2020r (Module) :

	def __init__ (self, params : Dict[str, Dict] = None) -> None :
		"""
		Doc
		"""

		super(Zrimec2020r, self).__init__()

		params = update_params(params = params)

		self.backbone = Zrimec2020(
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

		return 'zrimec2020r'

	@property
	def __str__ (self) -> str :
		"""
		Doc
		"""

		return 'Zrimec et. al (2020)'

	def forward (self, x : Tensor, v : Tensor = None) -> Tensor :
		"""
		Doc
		"""

		x = self.backbone(x, v)

		x = self.fc3(x)
		x = self.relu(x)

		return x

	def summary (self, batch_size : int, in_height : int, in_width : int, in_features : int, verbose : bool = False) -> ModelStatistics :
		""""
		Doc
		"""

		col_names = ['input_size', 'output_size', 'num_params']

		if verbose :
			col_names.extend(['kernel_size', 'mult_adds', 'trainable'])

		return torchinfo.summary(
			self,
			input_size = [(batch_size, in_height, in_width), (batch_size, in_features)],
			col_names  = col_names
		)

if __name__ == '__main__' :
	model = Zrimec2020r(params = {
		'other' : {
			'in_height'   : 4,
			'in_width'    : 2150,
			'in_features' : 64
		},
		'fc3' : {
			'features' : 8
		}
	})

	model.summary(batch_size = 64, in_height = 4, in_width = 2150, in_features = 64)
