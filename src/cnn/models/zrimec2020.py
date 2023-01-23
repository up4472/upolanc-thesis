from torch     import Tensor
from torch.nn  import BatchNorm1d
from torch.nn  import Conv1d
from torch.nn  import Dropout
from torch.nn  import Flatten
from torch.nn  import Linear
from torch.nn  import MaxPool1d
from torch.nn  import Module
from torch.nn  import ReLU
from torchinfo import ModelStatistics
from typing    import Any
from typing    import Dict

import torch
import torchinfo

from src.cnn.models._util import compute1d

def update_params (params : Dict[str, Any] = None) -> Dict[str, Any] :
	"""
	Doc
	"""

	default = {
		'other' : {
			'in_features' : 64,
			'in_width' : 2150,
			'in_height' : 4,
			'dropout' : 0.25
		},
		'conv1' : {
			'filters' : 32,
			'kernel' : 11,
			'padding' : 0,
			'dilation' : 1
		},
		'conv2' : {
			'filters' : 64,
			'kernel' : 11,
			'padding' : 5,
			'dilation' : 1
		},
		'conv3' : {
			'filters' : 128,
			'kernel' : 11,
			'padding' : 5,
			'dilation' : 1
		},
		'maxpool1' : {
			'kernel' : 5,
			'stride' : 5,
			'padding' : 2,
		},
		'maxpool2' : {
			'kernel' : 5,
			'stride' : 5,
			'padding' : 2,
		},
		'maxpool3' : {
			'kernel' : 5,
			'stride' : 5,
			'padding' : 2,
		},
		'fc1' : {
			'features' : 128,
		},
		'fc2' : {
			'features' : 64
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

			if padding != 0 and padding != (kernel - 1) // 2 :
				print(f'Problem with padding in [{layer}] : [{padding}] : [{kernel}]')

	return default

class Zrimec2020 (Module) :

	def __init__ (self, params : Dict[str, Any] = None) -> None :
		"""
		Doc
		"""

		super(Zrimec2020, self).__init__()

		params = update_params(
			params = params
		)

		self.params = update_params(
			params = params
		)

		self.conv1 = Conv1d(
			in_channels  = params['other']['in_height'],
			out_channels = params['conv1']['filters'],
			kernel_size  = params['conv1']['kernel'],
			padding      = params['conv1']['padding'],
			dilation     = params['conv1']['dilation']
		)

		self.conv2 = Conv1d(
			in_channels  = params['conv1']['filters'],
			out_channels = params['conv2']['filters'],
			kernel_size  = params['conv2']['kernel'],
			padding      = params['conv2']['padding'],
			dilation     = params['conv2']['dilation']
		)

		self.conv3 = Conv1d(
			in_channels  = params['conv2']['filters'],
			out_channels = params['conv3']['filters'],
			kernel_size  = params['conv3']['kernel'],
			padding      = params['conv3']['padding'],
			dilation     = params['conv3']['dilation']
		)

		self.bn1 = BatchNorm1d(num_features = params['conv1']['filters'])
		self.bn2 = BatchNorm1d(num_features = params['conv2']['filters'])
		self.bn3 = BatchNorm1d(num_features = params['conv3']['filters'])

		self.maxpool1 = MaxPool1d(
			kernel_size = params['maxpool1']['kernel'],
			stride      = params['maxpool1']['stride'],
			padding     = params['maxpool1']['padding']
		)

		self.maxpool2 = MaxPool1d(
			kernel_size = params['maxpool1']['kernel'],
			stride      = params['maxpool1']['stride'],
			padding     = params['maxpool1']['padding']
		)

		self.maxpool3 = MaxPool1d(
			kernel_size = params['maxpool1']['kernel'],
			stride      = params['maxpool1']['stride'],
			padding     = params['maxpool1']['padding']
		)

		self.flatten = Flatten()

		size = params['other']['in_width']
		size = compute1d(size = size, module = self.conv1)
		size = compute1d(size = size, module = self.maxpool1)
		size = compute1d(size = size, module = self.conv2)
		size = compute1d(size = size, module = self.maxpool2)
		size = compute1d(size = size, module = self.conv3)
		size = compute1d(size = size, module = self.maxpool3)

		size = size * params['conv3']['filters']     # flatten (channels)
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
		self.relu    = ReLU(inplace = False)

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

		if v is not None :
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
		'other' : {
			'in_height'   : 4,
			'in_width'    : 2150,
			'in_features' : 64
		},
		'fc2' : {
			'features' : 64
		}
	})

	model.summary(batch_size = 64, in_height = 4, in_width = 2150, in_features = 64)
