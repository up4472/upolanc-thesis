import multiprocessing
import numpy
import os
import pandas
import platform
import psutil
import random
import re
import seaborn
import subprocess
import torch

def set_numpy_format () -> None :
	"""
	Doc
	"""

	numpy.set_printoptions(
		suppress  = True,
		edgeitems = 25,
		linewidth = 150,
		formatter = {
			'float_kind' : '{: 7,.3f}'.format
		}
	)

def set_pandas_format () -> None :
	"""
	Doc
	"""

	pandas.set_option(
		'display.float_format',
		'{:.3f}'.format
	)

def set_plot_theme () -> None :
	"""
	Doc
	"""

	seaborn.set_theme()

def lock_random (seed : int = None, generate : bool = False) -> int :
	"""
	Doc
	"""

	if seed is None and generate :
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

	gpu_available = torch.cuda.is_available()
	gpu_count     = torch.cuda.device_count()
	gpu_name      = 'N/A'

	if gpu_available :
		gpu_name = torch.cuda.get_device_name()

	return {
		'platform/python'           : platform_python,
		'platform/system'           : platform_system,
		'platform/release'          : platform_release,
		'platform/version'          : platform_version,
		'platform/cpu/name'         : cpu_name,
		'platform/cpu/count'        : cpu_count,
		'platform/gpu/name'         : gpu_name,
		'platform/gpu/available'    : gpu_available,
		'platform/gpu/count'        : gpu_count,
		'platform/memory/total'     : memory_total,
		'platform/memory/available' : memory_available
	}
