from pandas import DataFrame
from typing import Any
from typing import Dict

import json
import pandas
import math
import numpy
import os

def flatten_dict (data : Dict[str, Any], prefix : str = None, sep : str = '/') -> Dict[str, Any] :
	"""
	Doc
	"""

	items = []

	for k, v in data.items() :
		if prefix is not None :
			nk = prefix + sep + k
		else :
			nk = k

		if isinstance(v, dict) :
			items.extend(flatten_dict(v, nk, sep).items())
		else :
			items.append((nk, v))

	return dict(items)

def recover_dataframe (logdir : str) -> DataFrame :
	"""
	Doc
	"""

	data = dict()

	for log in os.scandir(logdir) :
		if not os.path.isdir(log.path) :
			continue

		result = os.path.join(log.path, 'result.json')

		with open(result, mode = 'r') as handle :
			result = handle.readlines()[-1]
			result = json.loads(result)
			result = flatten_dict(result, None, '/')

		data[log.name] = result

	dataframe = DataFrame.from_dict(data, orient = 'index')
	dataframe = dataframe.reset_index(names = ['logdir'])
	dataframe = dataframe.replace('null', numpy.nan)

	return dataframe

def convert_json_to_dataframe (root : str, source_name : str, target_name : str) -> DataFrame :
	"""
	Doc
	"""

	target_name = os.path.join(root, target_name)
	source_name = os.path.join(root, source_name)

	if os.path.exists(target_name) :
		return pandas.read_csv(target_name)

	with open(source_name, mode = 'r') as handle :
		data = []

		for line in handle.readlines() :
			if '[' in line or ']' in line :
				continue

			if   '{' in line : item = {}
			elif '}' in line : data.append(item)
			else :
				line = line.split(':')

				key = line[0].strip().replace('"', '')
				val = line[1].strip().replace('"', '').replace(',', '')

				item[str(key)] = float(val)

	dataframe = DataFrame(data)
	dataframe.to_csv(target_name)

	return dataframe

def convert_bert_step_to_epoch (step : int, steps_per_epoch : int = 485, floor : bool = False) -> float :
	"""
	Doc
	"""

	epoch = step / steps_per_epoch

	if floor :
		epoch = math.floor(epoch)

	return epoch
