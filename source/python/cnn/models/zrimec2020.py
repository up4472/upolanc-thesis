from torch     import Tensor
from torch.nn  import BatchNorm1d
from torch.nn  import Conv1d
from torch.nn  import Dropout
from torch.nn  import Flatten
from torch.nn  import LeakyReLU
from torch.nn  import Linear
from torch.nn  import MaxPool1d
from torch.nn  import Module
from torchinfo import ModelStatistics
from typing    import Any
from typing    import Dict

import torch
import torchinfo

from source.python.cnn.models._util import compute1d

def _ensure_stride (params : Dict[str, Any]) -> Dict[str, Any] :
	"""
	Doc
	"""

	for key, value in params.items() :
		if key.endswith('stride') :
			params[key] = params[key.replace('stride', 'kernel')]

	return params

def _ensure_padding (params : Dict[str, Any]) -> Dict[str, Any] :
	"""
	Doc
	"""

	for key, value in params.items() :
		if key.endswith('padding') :
			padding = value
			kernel  = params[key.replace('padding', 'kernel')]

			if isinstance(padding, str) :
				padding = padding.lower()

				if   padding == 'none'  : padding = 0
				elif padding == 'same'  : padding = (kernel - 1) // 2
				elif padding == 'valid' : padding = (kernel - 1) // 2
				else : raise ValueError()

				params[key] = padding

			if padding != 0 and padding != (kernel - 1) // 2 :
				print(f'Problem with padding in [{key}] : [{padding}] : [{kernel}]')

	return params

def update_params (params : Dict[str, Any] = None) -> Dict[str, Any] :
	"""
	Doc
	"""

	default = {
		'model/input/channels' : 1,
		'model/input/height'   : 4,
		'model/input/width'    : 2150,
		'model/input/features' : 64,

		'model/dropout'   : 0.10,
		'model/leakyrelu' : 0.00,

		'model/conv1/filters'  : 64,
		'model/conv1/kernel'   : 11,
		'model/conv1/padding'  : 'none',
		'model/conv1/dilation' : 1,
		'model/conv2/filters'  : 64,
		'model/conv2/kernel'   : 11,
		'model/conv2/padding'  : 'none',
		'model/conv2/dilation' : 1,
		'model/conv3/filters'  : 128,
		'model/conv3/kernel'   : 11,
		'model/conv3/padding'  : 'same',
		'model/conv3/dilation' : 1,

		'model/maxpool1/kernel'  : 5,
		'model/maxpool1/stride'  : 5,
		'model/maxpool1/padding' : 'same',
		'model/maxpool2/kernel'  : 5,
		'model/maxpool2/stride'  : 5,
		'model/maxpool2/padding' : 'same',
		'model/maxpool3/kernel'  : 5,
		'model/maxpool3/stride'  : 5,
		'model/maxpool3/padding' : 'same',

		'model/fc1/features' : 128,
		'model/fc2/features' : 256,

		'model/features' : False
	}

	if params is None :
		return default

	for key, value in params.items() :
		if key.startswith('model') :
			default[key] = value

	default = _ensure_stride(params = default)
	default = _ensure_padding(params = default)

	return default

class Zrimec2020 (Module) :

	def __init__ (self, params : Dict[str, Any] = None) -> None :
		"""
		Doc
		"""

		super(Zrimec2020, self).__init__()

		self.params = update_params(
			params = params
		)

		self.conv1 = Conv1d(
			in_channels  = self.params['model/input/height'],
			out_channels = self.params['model/conv1/filters'],
			kernel_size  = self.params['model/conv1/kernel'],
			padding      = self.params['model/conv1/padding'],
			dilation     = self.params['model/conv1/dilation']
		)

		self.conv2 = Conv1d(
			in_channels  = self.params['model/conv1/filters'],
			out_channels = self.params['model/conv2/filters'],
			kernel_size  = self.params['model/conv2/kernel'],
			padding      = self.params['model/conv2/padding'],
			dilation     = self.params['model/conv2/dilation']
		)

		self.conv3 = Conv1d(
			in_channels  = self.params['model/conv2/filters'],
			out_channels = self.params['model/conv3/filters'],
			kernel_size  = self.params['model/conv3/kernel'],
			padding      = self.params['model/conv3/padding'],
			dilation     = self.params['model/conv3/dilation']
		)

		self.bn1 = BatchNorm1d(num_features = self.params['model/conv1/filters'])
		self.bn2 = BatchNorm1d(num_features = self.params['model/conv2/filters'])
		self.bn3 = BatchNorm1d(num_features = self.params['model/conv3/filters'])

		self.maxpool1 = MaxPool1d(
			kernel_size = self.params['model/maxpool1/kernel'],
			stride      = self.params['model/maxpool1/stride'],
			padding     = self.params['model/maxpool1/padding']
		)

		self.maxpool2 = MaxPool1d(
			kernel_size = self.params['model/maxpool2/kernel'],
			stride      = self.params['model/maxpool2/stride'],
			padding     = self.params['model/maxpool2/padding']
		)

		self.maxpool3 = MaxPool1d(
			kernel_size = self.params['model/maxpool3/kernel'],
			stride      = self.params['model/maxpool3/stride'],
			padding     = self.params['model/maxpool3/padding']
		)

		self.flatten = Flatten()

		size = self.params['model/input/width']
		size = compute1d(size = size, module = self.conv1)
		size = compute1d(size = size, module = self.maxpool1)
		size = compute1d(size = size, module = self.conv2)
		size = compute1d(size = size, module = self.maxpool2)
		size = compute1d(size = size, module = self.conv3)
		size = compute1d(size = size, module = self.maxpool3)

		size = size * self.params['model/conv3/filters']

		if self.params['model/features'] :
			size = size + self.params['model/input/features']

		self.fc1 = Linear(
			in_features  = size,
			out_features = self.params['model/fc1/features']
		)

		self.fc2 = Linear(
			in_features  = self.params['model/fc1/features'],
			out_features = self.params['model/fc2/features']
		)

		self.dropout = Dropout(
			p       = self.params['model/dropout'],
			inplace = False
		)

		self.relu = LeakyReLU(
			negative_slope = self.params['model/leakyrelu'],
			inplace        = False
		)

	@property
	def __name__ (self) -> str :
		"""
		Doc
		"""

		return 'zrimec2020'

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

		x = self.conv1(x)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.dropout(x)
		x = self.maxpool1(x)

		x = self.conv2(x)
		x = self.relu(x)
		x = self.bn2(x)
		x = self.dropout(x)
		x = self.maxpool2(x)

		x = self.conv3(x)
		x = self.relu(x)
		x = self.bn3(x)
		x = self.dropout(x)
		x = self.maxpool3(x)

		x = self.flatten(x)

		if self.params['model/features'] and v is not None :
			x = torch.hstack((x, v))

		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)

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
	model = Zrimec2020(params = {
		'model/input/height'   : 4,
		'model/input/width'    : 2150,
		'model/input/features' : 64,
		'model/fc2/features'   : 64
	})

	model.summary(batch_size = 64, in_height = 4, in_width = 2150, in_features = 64)
