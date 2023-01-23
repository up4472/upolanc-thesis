import numpy
import random
import torch

def lock_random (seed : int = None, generate_seed : bool = False) -> int :
	"""
	Doc
	"""

	if seed is None and generate_seed :
		seed = random.randint(1, 1_000_000_000_000)

	if seed is not None :
		torch.manual_seed(seed)
		numpy.random.seed(seed)
		random.seed(seed)

	return seed

def get_device (only_cpu : bool = False) -> torch.device :
	"""
	Doc
	"""

	device = torch.device('cpu')

	if not only_cpu and torch.cuda.is_available() :
		print(f'Graphic devices : {torch.cuda.device_count()}')

		device = torch.device('cuda')

	print(f'Selected device : {device}')

	return device
