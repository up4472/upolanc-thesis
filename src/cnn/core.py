import multiprocessing
import numpy
import os
import platform
import psutil
import random
import re
import subprocess
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

	memory = psutil.virtual_memory()

	memory_factor     = 2 ** 30
	memory_total     = '{:.3f} GB'.format(memory.total     / memory_factor)
	memory_available = '{:.3f} GB'.format(memory.available / memory_factor)

	platform_python  = platform.python_version()
	platform_system  = platform.system()
	platform_release = platform.release()
	platform_version = platform.version()

	cpu_name  = 'N/A'
	cpu_count = multiprocessing.cpu_count()

	if platform_system == 'Linux' :
		proc = subprocess.check_output('cat /proc/cpuinfo', shell = True)
		proc = proc.decode().strip()

		for line in proc.split('\n') :
			if 'model name' in line :
				cpu_name = re.sub('.*model name.*:', '', line, 1)
				cpu_name = cpu_name.strip()

				break

		cpu_count = len(os.sched_getaffinity(0))

	cuda_available = torch.cuda.is_available()
	cuda_devices   = torch.cuda.device_count()
	cuda_name      = 'N/A'

	if cuda_available :
		cuda_name = torch.cuda.get_device_name()

	return {
		'platform_python'  : platform_python,
		'platform_system'  : platform_system,
		'platform_release' : platform_release,
		'platform_version' : platform_version,
		'cpu_name'         : cpu_name,
		'cpu_count'        : cpu_count,
		'cuda_name'        : cuda_name,
		'cuda_available'   : cuda_available,
		'cuda_devices'     : cuda_devices,
		'memory_total'     : memory_total,
		'memory_available' : memory_available
	}
