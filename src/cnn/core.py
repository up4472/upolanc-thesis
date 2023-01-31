import multiprocessing
import numpy
import platform
import random
import torch

def lock_random (seed : int = None, generate_seed : bool = False) -> int :
	"""
	Doc
	"""

	if seed is None and generate_seed :
		seed = random.randint(1, 1_073_741_824)

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
		device = torch.device('cuda')

	return device

def get_system_info () :
	"""
	Doc
	"""

	platform_system  = platform.system()
	platform_release = platform.release()
	platform_version = platform.version()

	gpu_count = torch.cuda.device_count()
	cpu_count = multiprocessing.cpu_count()

	return {
		'platform_system'  : platform_system,
		'platform_release' : platform_release,
		'platform_version' : platform_version,
		'cpu_count'        : cpu_count,
		'gpu_count'        : gpu_count
	}
