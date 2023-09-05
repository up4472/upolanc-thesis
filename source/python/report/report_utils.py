from pandas import DataFrame
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import json
import pandas
import math
import numpy
import os

from source.python.report.report_constants import BERT_ARCH
from source.python.report.report_constants import BERT_LAYERS
from source.python.report.report_constants import BERT_OUTPUT
from source.python.report.report_constants import BERT_POOLER
from source.python.report.report_constants import COLORS
from source.python.report.report_constants import FEATURES
from source.python.report.report_constants import FILTERS
from source.python.report.report_constants import KMERS
from source.python.report.report_constants import OPTIMIZERS
from source.python.report.report_constants import SEQUENCES
from source.python.report.report_constants import TARGETS

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

				if   val is None   : val = numpy.nan
				elif val == 'None' : val = numpy.nan
				elif val == 'none' : val = numpy.nan

				item[str(key)] = float(val)

	dataframe = DataFrame(data)
	dataframe.to_csv(target_name)

	return dataframe

def convert_bert_group_to_color (name : str, groupby : str = None) -> Optional[str] :
	"""
	Doc
	"""

	if groupby is None :
		return None

	tokens = name.split('-')

	if   groupby == 'output'    : array = BERT_OUTPUT; token = tokens[0]
	elif groupby == 'arch'      : array = BERT_ARCH;   token = tokens[1]
	elif groupby == 'pooler'    : array = BERT_POOLER; token = tokens[2]
	elif groupby == 'layers'    : array = BERT_LAYERS; token = tokens[3]
	elif groupby == 'kmer'      : array = KMERS;       token = tokens[4]
	elif groupby == 'features'  : array = FEATURES;    token = tokens[5]
	elif groupby == 'sequence'  : array = SEQUENCES;   token = tokens[6]
	elif groupby == 'optimizer' : array = OPTIMIZERS;  token = tokens[7]
	elif groupby == 'filters'   : array = FILTERS;     token = tokens[8]
	elif groupby == 'target'    : array = TARGETS;     token = '-'.join(tokens[10:])
	else : return None

	return COLORS[array.index(token)]

def convert_bert_step_to_epoch (step : int, steps_per_epoch : Union[int, float] = 485, floor : bool = False) -> Union[int, float] :
	"""
	Doc
	"""

	epoch = step / steps_per_epoch

	if floor :
		return math.floor(epoch)

	return epoch

def convert_bert_name (name : str, style : str = None, index : int = None) -> str :
	"""
	Doc
	"""

	if style is None : return name

	style = style.lower()

	if style == 'none'     : return name
	if style == 'original' : return name

	tokens = name.split('-')

	arch      = tokens[1]
	pooler    = tokens[2]
	kmer      = tokens[4]
	sequence  = tokens[6]
	filterid  = tokens[8][1:]

	if len(tokens) == 13 : target = '{}-{}-{}'.format(tokens[-3] ,tokens[-2], tokens[-1])
	else                 : target = '{}-{}'.format(tokens[-2], tokens[-1])

	if   pooler == 'v1' : pooler = 'first'
	elif pooler == 'v2' : pooler = 'mean'

	if   sequence == 'po0512' : sequence = 'promoter 512 bp'
	elif sequence == 'po4096' : sequence = 'promoter 4096 bp'
	elif sequence == 'po5000' : sequence = 'promoter 5000 bp'
	elif sequence == 'pu4096' : sequence = 'promoter + 5\'utr 4096 bp'
	elif sequence == 'pu5000' : sequence = 'promoter + 5\'utr 5000 bp'
	elif sequence == 'tf2150' : sequence = 'transcript 2150 bp'
	elif sequence == 'tf6150' : sequence = 'transcript 6150 bp'

	if style == 'sequence'            : return sequence
	if style == 'target'              : return target
	if style == 'pooler'              : return pooler
	if style == 'architecture'        : return arch
	if style == 'kmer'                : return '{}-mer'.format(kmer)
	if style == 'pooler-architecture' : return '{}-{}'.format(arch, pooler)
	if style == 'filter'              : return 'filter {}'.format(filterid)
	if style == 'index'               : return '{:02d} : {}'.format(index, name)

	return name

def convert_errlog_to_dataframe (root : str, source_name : str, target_name : str, step_size : int = 100) -> DataFrame :
	"""
	Doc
	"""

	target_name = os.path.join(root, target_name)
	source_name = os.path.join(root, source_name)

	if os.path.exists(target_name) :
		return pandas.read_csv(target_name)

	metrics = ['r2', 'mae', 'mape', 'max_error', 'mse']
	scores  = dict()

	with open(source_name, mode = 'r') as handle :
		for line in handle.readlines() :
			if 'INFO' not in line :
				continue

			for metric in metrics :
				metric = metric.lower()
				string = metric + ' = '

				if string not in line :
					continue

				index = line.index(string) + len(string)
				score = line[index:].strip()

				if score == 'None' or score == 'none' :
					score = float(0.0)
				else :
					score = float(score)

				if metric not in scores.keys() :
					scores[metric] = list()

				scores[metric].append(score)

	record = list()
	length = len(scores['r2'])

	for index in range(length) :
		record.append({
			'eval_r2'        : scores['r2'][index],
			'eval_max_error' : scores['max_error'][index],
			'eval_mape'      : scores['mape'][index],
			'eval_mae'       : scores['mae'][index],
			'eval_mse'       : scores['mse'][index],
			'learning_rate'  : numpy.nan,
			'loss'           : numpy.nan,
			'step'           : int(step_size * (1 + index))
		})

	dataframe = DataFrame.from_records(record)
	dataframe.to_csv(target_name)

	return dataframe
