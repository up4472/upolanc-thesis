from pandas import DataFrame
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import pandas

from source.python.report.report_format import format_bert_data_dataframe
from source.python.report.report_format import format_data_tune_dataframe
from source.python.report.report_format import format_cnn_tune_dataframe
from source.python.report.report_format import format_feature_tune_dataframe
from source.python.report.report_utils  import convert_bert_step_to_epoch

TUNE_FEATURE = 0
TUNE_CNN     = 1
TUNE_DATA    = 2

def concat_tune_reports_format (reports : Dict, mode : str, tune_type : int, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	data = None

	if   tune_type == TUNE_FEATURE : formatter = format_feature_tune_dataframe
	elif tune_type == TUNE_CNN     : formatter = format_cnn_tune_dataframe
	elif tune_type == TUNE_DATA    : formatter = format_data_tune_dataframe
	else                           : raise ValueError()

	if reports is None         : return None
	if reports[mode] is None   : return None
	if len(reports[mode]) == 0 : return None

	for key, dataframe in reports[mode].items() :
		keys = key.split('-')

		if tune_type == TUNE_CNN :
			arch     = keys[0]
			sequence = keys[1]
			filters  = keys[2]
			trials   = keys[3] # noqa
			epochs   = keys[4] # noqa
			features = keys[5]
			target0  = keys[6]
			target1  = keys[7] if len(keys) >= 8 else None
			target2  = keys[8] if len(keys) >= 9 else None

			dataframe = dataframe.copy()
			dataframe.insert(0, 'Model',    arch)
			dataframe.insert(1, 'Sequence', sequence)
			dataframe.insert(2, 'Filter',   filters)
			dataframe.insert(3, 'Features', features)
			dataframe.insert(4, 'Target0',  target0)
			dataframe.insert(5, 'Target1',  target1)
			dataframe.insert(5, 'Target2',  target2)

		if tune_type == TUNE_DATA :
			arch     = keys[0]
			sequence = keys[1]
			filters  = keys[2]
			target0  = keys[5]
			target1  = keys[6] if len(keys) >= 7 else None
			target2  = keys[7] if len(keys) >= 8 else None

			dataframe = dataframe.copy()
			dataframe.insert(0, 'Model',    arch)
			dataframe.insert(1, 'Sequence', sequence)
			dataframe.insert(2, 'Filter',   filters)
			dataframe.insert(4, 'Target0',  target0)
			dataframe.insert(5, 'Target1',  target1)
			dataframe.insert(5, 'Target2',  target2)

		if tune_type == TUNE_FEATURE :
			dataframe = dataframe.copy()

			dataframe.insert(0, 'Target0', None)
			dataframe.insert(1, 'Target1', None)
			dataframe.insert(2, 'Target2', None)

		if data is None :
			data = dataframe
		else :
			data = pandas.concat((data, dataframe))

	if   mode == 'regression'     : sort_by = 'valid_r2'
	elif mode == 'classification' : sort_by = 'valid_accuracy'
	else                          : raise ValueError()

	data = data.sort_values(sort_by, ascending = False, na_position = 'last')
	data = data.reset_index()

	data = formatter(
		dataframe = data,
		mode      = mode
	).head(n = n)

	if tune_type == TUNE_FEATURE :
		target = data['Target'].tolist()
		target = [item.split('-') for item in target]

		data = data.drop(columns = ['Target'])

		data['Target0'] = [item[0] if len(item) >= 1 else None for item in target]
		data['Target1'] = [item[1] if len(item) >= 2 else None for item in target]
		data['Target2'] = [item[2] if len(item) >= 3 else None for item in target]

	return data

def concat_cnn_tune_reports (reports : Dict, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	return concat_tune_reports_format(
		reports   = reports,
		mode      = mode,
		n         = n,
		tune_type = TUNE_CNN
	)

def concat_data_tune_reports (reports : Dict, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	return concat_tune_reports_format(
		reports   = reports,
		mode      = mode,
		n         = n,
		tune_type = TUNE_DATA
	)

def concat_feature_tune_reports (reports : Dict, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	return concat_tune_reports_format(
		reports   = reports,
		mode      = mode,
		n         = n,
		tune_type = TUNE_FEATURE
	)

def concat_bert_reports (data : Dict[str, Any], mode : str, metric : str, ascending : bool = False, steps_per_epoch : Union[int, float] = 485) -> Optional[DataFrame] :
	"""
	Doc
	"""

	if data is None         : return None
	if data[mode] is None   : return None
	if len(data[mode]) == 0 : return None

	array = list()

	for key, dataframe in data[mode].items() :
		dataframe = dataframe.sort_values(metric, ascending = ascending)
		item      = dataframe.iloc[0, :].to_dict()
		tokens    = key.split('-')

		bert_arch     = tokens[0]
		bert_type     = tokens[1]
		bert_pooler   = tokens[2]
		bert_layer    = tokens[3]
		bert_kmer     = tokens[4]
		bert_feature  = tokens[5]
		bert_sequence = tokens[6]
		bert_optim    = tokens[7]
		bert_filter   = tokens[8]
		bert_target0  = tokens[10]
		bert_target1  = tokens[11] if len(tokens) >= 12 else None
		bert_target2  = tokens[12] if len(tokens) >= 13 else None

		if   bert_pooler == 'v1' : bert_pooler = 'def'
		elif bert_pooler == 'v2' : bert_pooler = 'dna'

		item['Mode']      = mode
		item['Arch']      = bert_arch
		item['Pooler']    = bert_pooler
		item['Type']      = bert_type
		item['Layer']     = bert_layer
		item['Kmer']      = bert_kmer
		item['Feature']   = bert_feature
		item['Filter']    = bert_filter
		item['Sequence']  = bert_sequence
		item['Optimizer'] = bert_optim
		item['Target0']   = bert_target0
		item['Target1']   = bert_target1
		item['Target2']   = bert_target2

		if item['Target2'] == 'explode' : sitr = int(steps_per_epoch * 5)
		else                            : sitr = int(steps_per_epoch)

		item['Steps']     = dataframe['step'].max()
		item['Epochs']    = convert_bert_step_to_epoch(
			step            = item['Steps'],
			steps_per_epoch = sitr,
			floor           = False
		)

		item['Step']  = item['step']
		item['Epoch'] = convert_bert_step_to_epoch(
			step            = item['Step'],
			steps_per_epoch = sitr,
			floor           = False
		)

		array.append(item)

	return format_bert_data_dataframe(
		dataframe = DataFrame(array),
		mode      = mode
	)
