from torch     import Tensor
from torch.nn  import Linear
from torch.nn  import Module
from torch.nn  import ModuleList
from torchinfo import ModelStatistics
from typing    import Any
from typing    import Dict

import torch
import torchinfo

from src.cnn.models.zrimec2020 import Zrimec2020
from src.cnn.models.zrimec2020 import update_params

class Zrimec2020c (Module) :

	def __init__ (self, params : Dict[str, Any] = None) -> None :
		"""
		Doc
		"""

		super(Zrimec2020c, self).__init__()

		params = update_params(params = params)

		self.backbone = Zrimec2020(
			params = params
		)

		self.heads = ModuleList([
			Linear(
				in_features  = params['fc1']['features'],
				out_features = params['fc2']['features']
			)
			for _ in range(params['fc2']['heads'])
		])

	@property
	def __name__ (self) -> str :
		"""
		Doc
		"""

		return 'zrimec2020c'

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

		x = torch.stack([
			head(x) for head in self.heads
		], dim = 2)

		return x

	def summary (self, batch_size: int, in_height: int, in_width: int, in_features : int, verbose : bool = False) -> ModelStatistics :
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
	model = Zrimec2020c(params = {
		'other' : {
			'in_height'   : 4,
			'in_width'    : 2150,
			'in_features' : 64
		},
		'fc2' : {
			'heads'    : 8,
			'features' : 5
		}
	})

	model.summary(batch_size = 64, in_height = 4, in_width = 2150, in_features = 64)
