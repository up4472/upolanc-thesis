from torch     import Tensor
from torch.nn  import Linear
from torch.nn  import Module
from torch.nn  import ModuleList
from torchinfo import ModelStatistics
from typing    import Any
from typing    import Dict

import torch
import torchinfo

from source.python.cnn.models.zrimec2020 import Zrimec2020
from source.python.cnn.models.zrimec2020 import update_params

class Zrimec2020c (Module) :

	def __init__ (self, params : Dict[str, Any] = None) -> None :
		"""
		Doc
		"""

		super(Zrimec2020c, self).__init__()

		self.params = update_params(
			params = params
		)

		self.backbone = Zrimec2020(
			params = self.params
		)

		self.fc3 = ModuleList([
			Linear(
				in_features = self.params['model/fc2/features'],
				out_features = self.params['model/fc3/features']
			)
			for _ in range(self.params['model/fc3/heads'])
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
			fc(x) for fc in self.fc3
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
		'model/input/height'   : 4,
		'model/input/width'    : 2150,
		'model/input/features' : 64,
		'model/fc3/features'   : 5,
		'model/fc3/heads'      : 8
	})

	model.summary(batch_size = 64, in_height = 4, in_width = 2150, in_features = 64)
