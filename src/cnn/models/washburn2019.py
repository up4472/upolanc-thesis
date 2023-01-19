from torch     import Tensor
from torch.nn  import Conv2d
from torch.nn  import Dropout
from torch.nn  import Flatten
from torch.nn  import Linear
from torch.nn  import MaxPool2d
from torch.nn  import Module
from torch.nn  import ReLU
from torchinfo import ModelStatistics
from typing    import Any
from typing    import Dict

import torch
import torchinfo

from src.cnn.models._util import compute2d

def update_params (params : Dict[str, Any] = None) -> Dict[str, Any] :
	"""
	Doc
	"""

	default = {
		'other' : {
			'in_features' : 64,
			'in_channels' : 1,
			'in_width'    : 2150,
			'in_height'   : 4,
			'dropout'     : 0.25
		},
		'conv1' : {
			'filters'  : 64,
			'kernel'   : (4, 9),
			'padding'  : (0, 0),
			'dilation' : 1
		},
		'conv2' : {
			'filters'  : 64,
			'kernel'   : (1, 9),
			'padding'  : (0, 4),
			'dilation' : 1
		},
		'conv3' : {
			'filters'  : 128,
			'kernel'   : (1, 9),
			'padding'  : (0, 4),
			'dilation' : 1
		},
		'conv4' : {
			'filters'  : 128,
			'kernel'   : (1, 9),
			'padding'  : (0, 4),
			'dilation' : 1
		},
		'conv5' : {
			'filters'  : 64,
			'kernel'   : (1, 9),
			'padding'  : (0, 4),
			'dilation' : 1
		},
		'conv6' : {
			'filters'  : 64,
			'kernel'   : (1, 9),
			'padding'  : (0, 4),
			'dilation' : 1
		},
		'maxpool1' : {
			'kernel'  : (1, 3),
			'stride'  : (1, 3),
			'padding' : (0, 1),
		},
		'maxpool2' : {
			'kernel'  : (1, 3),
			'stride'  : (1, 3),
			'padding' : (0, 1),
		},
		'maxpool3' : {
			'kernel'  : (1, 3),
			'stride'  : (1, 3),
			'padding' : (0, 1),
		},
		'fc1' : {
			'features' : 128,
		},
		'fc2' : {
			'features' : 32,
		}
	}

	if params is None :
		return default

	for key, value in params.items() :
		if key in default.keys() :
			default[key].update(value)
		else :
			default[key] = value

	for layer, config in default.items() :
		if 'kernel' in config.keys() :
			kernel  = config['kernel']
			padding = config['padding']

			p0, p1 = padding
			k0, k1 = kernel

			if p0 != 0 and p0 != (k0 - 1) // 2 :
				print(f'Problem with padding in [{layer}] : [{p0}-{p1}] : [{k0}-{k1}]')
			if p1 != 0 and p1 != (k1 - 1) // 2 :
				print(f'Problem with padding in [{layer}] : [{p0}-{p1}] : [{k0}-{k1}]')

	return default

class Washburn2019 (Module) :

	def __init__ (self, params : Dict[str, Any] = None) -> None :
		"""
		Doc
		"""

		super(Washburn2019, self).__init__()

		params = update_params(params = params)

		self.conv1 = Conv2d(
			in_channels  = params['other']['in_channels'],
			out_channels = params['conv1']['filters'],
			kernel_size  = params['conv1']['kernel'],
			padding      = params['conv1']['padding'],
			dilation     = params['conv1']['dilation']
		)

		self.conv2 = Conv2d(
			in_channels  = params['conv1']['filters'],
			out_channels = params['conv2']['filters'],
			kernel_size  = params['conv2']['kernel'],
			padding      = params['conv2']['padding'],
			dilation     = params['conv2']['dilation']
		)

		self.conv3 = Conv2d(
			in_channels  = params['conv2']['filters'],
			out_channels = params['conv3']['filters'],
			kernel_size  = params['conv3']['kernel'],
			padding      = params['conv3']['padding'],
			dilation     = params['conv3']['dilation']
		)

		self.conv4 = Conv2d(
			in_channels  = params['conv3']['filters'],
			out_channels = params['conv4']['filters'],
			kernel_size  = params['conv4']['kernel'],
			padding      = params['conv4']['padding'],
			dilation     = params['conv4']['dilation']
		)

		self.conv5 = Conv2d(
			in_channels  = params['conv4']['filters'],
			out_channels = params['conv5']['filters'],
			kernel_size  = params['conv5']['kernel'],
			padding      = params['conv5']['padding'],
			dilation     = params['conv5']['dilation']
		)

		self.conv6 = Conv2d(
			in_channels  = params['conv5']['filters'],
			out_channels = params['conv6']['filters'],
			kernel_size  = params['conv6']['kernel'],
			padding      = params['conv6']['padding'],
			dilation     = params['conv6']['dilation']
		)

		self.maxpool1 = MaxPool2d(
			kernel_size = params['maxpool1']['kernel'],
			stride      = params['maxpool1']['stride'],
			padding     = params['maxpool1']['padding']
		)

		self.maxpool2 = MaxPool2d(
			kernel_size = params['maxpool2']['kernel'],
			stride      = params['maxpool2']['stride'],
			padding     = params['maxpool2']['padding']
		)

		self.maxpool3 = MaxPool2d(
			kernel_size = params['maxpool3']['kernel'],
			stride      = params['maxpool3']['stride'],
			padding     = params['maxpool3']['padding']
		)

		self.flatten = Flatten()

		size = (
			params['other']['in_height'],
			params['other']['in_width']
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
		size = size * params['conv6']['filters']     # flatten (channels)
		size = size + params['other']['in_features'] # injects (hstack)

		self.fc1 = Linear(
			in_features  = size,
			out_features = params['fc1']['features']
		)

		self.fc2 = Linear(
			in_features  = params['fc1']['features'],
			out_features = params['fc2']['features']
		)

		self.dropout = Dropout(params['other']['dropout'])
		self.relu = ReLU(inplace = False)

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
		'other' : {
			'in_channels' : 1,
			'in_height'   : 4,
			'in_width'    : 2150,
			'in_features' : 64
		},
		'fc2' : {
			'features' : 32
		}
	})

	model.summary(batch_size = 64, in_channels = 1, in_height = 4, in_width = 2150, in_features = 64)
