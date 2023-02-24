from torch     import Tensor
from torch.nn  import BatchNorm2d
from torch.nn  import Conv2d
from torch.nn  import Dropout
from torch.nn  import Flatten
from torch.nn  import LeakyReLU
from torch.nn  import Linear
from torch.nn  import MaxPool2d
from torch.nn  import Module
from torchinfo import ModelStatistics
from typing    import Any
from typing    import Dict

import torch
import torchinfo

from source.python.cnn.models._util import compute2d

def _ensure_kernel (params : Dict[str, Any]) -> Dict[str, Any] :
	"""
	Doc
	"""

	for key, value in params.items() :
		if key.endswith('kernel') and isinstance(value, int) :
			if key == 'model/conv1/kernel' :
				params[key] = [4, value]
			else :
				params[key] = [1, value]
		if key.endswith('stride') and isinstance(value, int) :
			params[key] = [1, value]

	return params

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

				if   padding == 'none'  : padding = [0, 0]
				elif padding == 'same'  : padding = [(kernel[0] - 1) // 2, (kernel[1] - 1) // 2]
				elif padding == 'valid' : padding = [(kernel[0] - 1) // 2, (kernel[1] - 1) // 2]
				else : raise ValueError()

				params[key] = padding

			p0, p1 = padding
			k0, k1 = kernel

			if p0 != 0 and p0 != (k0 - 1) // 2 :
				print(f'Problem with padding in [{key}] : [{p0}-{p1}] : [{k0}-{k1}]')
			if p1 != 0 and p1 != (k1 - 1) // 2 :
				print(f'Problem with padding in [{key}] : [{p0}-{p1}] : [{k0}-{k1}]')

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
		'model/conv1/kernel'   : [4, 9],
		'model/conv1/padding'  : 'same',
		'model/conv1/dilation' : 1,
		'model/conv2/filters'  : 64,
		'model/conv2/kernel'   : [1, 9],
		'model/conv2/padding'  : 'same',
		'model/conv2/dilation' : 1,
		'model/conv3/filters'  : 128,
		'model/conv3/kernel'   : [1, 9],
		'model/conv3/padding'  : 'same',
		'model/conv3/dilation' : 1,
		'model/conv4/filters'  : 128,
		'model/conv4/kernel'   : [1, 9],
		'model/conv4/padding'  : 'same',
		'model/conv4/dilation' : 1,
		'model/conv5/filters'  : 64,
		'model/conv5/kernel'   : [1, 9],
		'model/conv5/padding'  : 'same',
		'model/conv5/dilation' : 1,
		'model/conv6/filters'  : 64,
		'model/conv6/kernel'   : [1, 9],
		'model/conv6/padding'  : 'same',
		'model/conv6/dilation' : 1,

		'model/maxpool1/kernel'  : [1, 3],
		'model/maxpool1/stride'  : [1, 3],
		'model/maxpool1/padding' : 'same',
		'model/maxpool2/kernel'  : [1, 3],
		'model/maxpool2/stride'  : [1, 3],
		'model/maxpool2/padding' : 'same',
		'model/maxpool3/kernel'  : [1, 3],
		'model/maxpool3/stride'  : [1, 3],
		'model/maxpool3/padding' : 'same',

		'model/fc1/features' : 256,
		'model/fc2/features' : 128,
	}

	if params is None :
		return default

	for key, value in params.items() :
		if key.startswith('model') :
			default[key] = value

	default = _ensure_kernel(params = default)
	default = _ensure_stride(params = default)
	default = _ensure_padding(params = default)

	return default

class Washburn2019 (Module) :

	def __init__ (self, params : Dict[str, Any] = None) -> None :
		"""
		Doc
		"""

		super(Washburn2019, self).__init__()

		params = update_params(params = params)

		self.conv1 = Conv2d(
			in_channels  = params['model/input/channels'],
			out_channels = params['model/conv1/filters'],
			kernel_size  = params['model/conv1/kernel'],
			padding      = params['model/conv1/padding'],
			dilation     = params['model/conv1/dilation']
		)

		self.conv2 = Conv2d(
			in_channels  = params['model/conv1/filters'],
			out_channels = params['model/conv2/filters'],
			kernel_size  = params['model/conv2/kernel'],
			padding      = params['model/conv2/padding'],
			dilation     = params['model/conv2/dilation']
		)

		self.conv3 = Conv2d(
			in_channels  = params['model/conv2/filters'],
			out_channels = params['model/conv3/filters'],
			kernel_size  = params['model/conv3/kernel'],
			padding      = params['model/conv3/padding'],
			dilation     = params['model/conv3/dilation']
		)

		self.conv4 = Conv2d(
			in_channels  = params['model/conv3/filters'],
			out_channels = params['model/conv4/filters'],
			kernel_size  = params['model/conv4/kernel'],
			padding      = params['model/conv4/padding'],
			dilation     = params['model/conv4/dilation']
		)

		self.conv5 = Conv2d(
			in_channels  = params['model/conv4/filters'],
			out_channels = params['model/conv5/filters'],
			kernel_size  = params['model/conv5/kernel'],
			padding      = params['model/conv5/padding'],
			dilation     = params['model/conv5/dilation']
		)

		self.conv6 = Conv2d(
			in_channels  = params['model/conv5/filters'],
			out_channels = params['model/conv6/filters'],
			kernel_size  = params['model/conv6/kernel'],
			padding      = params['model/conv6/padding'],
			dilation     = params['model/conv6/dilation']
		)

		self.bn1 = BatchNorm2d(num_features = params['model/conv1/filters'])
		self.bn2 = BatchNorm2d(num_features = params['model/conv2/filters'])
		self.bn3 = BatchNorm2d(num_features = params['model/conv3/filters'])
		self.bn4 = BatchNorm2d(num_features = params['model/conv4/filters'])
		self.bn5 = BatchNorm2d(num_features = params['model/conv5/filters'])
		self.bn6 = BatchNorm2d(num_features = params['model/conv6/filters'])

		self.maxpool1 = MaxPool2d(
			kernel_size = params['model/maxpool1/kernel'],
			stride      = params['model/maxpool1/stride'],
			padding     = params['model/maxpool1/padding']
		)

		self.maxpool2 = MaxPool2d(
			kernel_size = params['model/maxpool2/kernel'],
			stride      = params['model/maxpool2/stride'],
			padding     = params['model/maxpool2/padding']
		)

		self.maxpool3 = MaxPool2d(
			kernel_size = params['model/maxpool3/kernel'],
			stride      = params['model/maxpool3/stride'],
			padding     = params['model/maxpool3/padding']
		)

		self.flatten = Flatten()

		size = (
			params['model/input/height'],
			params['model/input/width']
		)

		size = compute2d(size = size, module = self.conv1)
		size = compute2d(size = size, module = self.conv2)
		size = compute2d(size = size, module = self.maxpool1)
		size = compute2d(size = size, module = self.conv3)
		size = compute2d(size = size, module = self.conv4)
		size = compute2d(size = size, module = self.maxpool2)
		size = compute2d(size = size, module = self.conv5)
		size = compute2d(size = size, module = self.conv6)
		size = compute2d(size = size, module = self.maxpool3)

		size = size[0] * size[1]                     # flatten (dims)
		size = size * params['model/conv6/filters']  # flatten (channels)
		size = size + params['model/input/features'] # injects (hstack)

		self.fc1 = Linear(
			in_features  = size,
			out_features = params['model/fc1/features']
		)

		self.fc2 = Linear(
			in_features  = params['model/fc1/features'],
			out_features = params['model/fc2/features']
		)

		self.dropout = Dropout(
			p       = params['model/dropout'],
			inplace = False
		)

		self.relu = LeakyReLU(
			negative_slope = params['model/leakyrelu'],
			inplace        = False
		)

	@property
	def __name__ (self) -> str :
		"""
		Doc
		"""

		return 'washburn2019'

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

		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)

		x = self.maxpool1(x)
		x = self.dropout(x)

		x = self.conv3(x)
		x = self.relu(x)
		x = self.conv4(x)
		x = self.relu(x)

		x = self.maxpool2(x)
		x = self.dropout(x)

		x = self.conv5(x)
		x = self.relu(x)
		x = self.conv6(x)
		x = self.relu(x)

		x = self.maxpool3(x)
		x = self.dropout(x)

		x = self.flatten(x)

		if v is not None :
			x = torch.hstack((x, v))

		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc2(x)
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
	model = Washburn2019(params = {
		'model/input/channels' : 1,
		'model/input/height'   : 4,
		'model/input/width'    : 2150,
		'model/input/features' : 64,
		'model/fc2/features'   : 32
	})

	model.summary(batch_size = 64, in_channels = 1, in_height = 4, in_width = 2150, in_features = 64)
