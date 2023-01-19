from torch.nn import AvgPool1d
from torch.nn import AvgPool2d
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import MaxPool1d
from torch.nn import MaxPool2d
from torch.nn import Module
from typing   import Tuple

import math

def compute2d (size : Tuple[int, int], module : Module) -> Tuple[int, int] :
	"""
	Doc
	"""

	h = size[0]
	w = size[1]

	if isinstance(module, Conv2d) :
		dilation = module.dilation
		padding  = module.padding
		kernel   = module.kernel_size
		stride   = module.stride

		if isinstance(padding, str) :
			raise ValueError()

		return (
			math.floor(1 + (h + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0]),
			math.floor(1 + (w + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1])
		)

	if isinstance(module, (MaxPool2d, AvgPool2d)) :
		dilation = module.dilation
		padding  = module.padding
		kernel   = module.kernel_size
		stride   = module.stride

		if isinstance(dilation, int) : dilation = (dilation, dilation)
		if isinstance(padding, int) : padding = (padding, padding)
		if isinstance(kernel, int) : kernel = (kernel, kernel)
		if isinstance(stride, int) : stride = (stride, stride)

		return (
			math.floor(1 + (h + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0]),
			math.floor(1 + (w + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1])
		)

	return h, w

def compute1d (size : int, module : Module) -> int :
	"""
	Doc
	"""

	if isinstance(module, Conv1d) :
		dilation = module.dilation
		padding  = module.padding
		kernel   = module.kernel_size
		stride   = module.stride

		if isinstance(padding, str) :
			raise ValueError()

		return math.floor(1 + (size + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0])

	if isinstance(module, (MaxPool1d, AvgPool1d)) :
		dilation = module.dilation
		padding  = module.padding
		kernel   = module.kernel_size
		stride   = module.stride

		if isinstance(dilation, tuple) : dilation = dilation[0]
		if isinstance(padding, tuple) : padding = padding[0]
		if isinstance(kernel, tuple) : kernel = kernel[0]
		if isinstance(stride, tuple) : stride = stride[0]

		return math.floor(1 + (size + 2 * padding - dilation * (kernel - 1) - 1) / stride)

	return size
