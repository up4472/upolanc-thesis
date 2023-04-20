from pandas import DataFrame
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import pandas

from source.python.report.report_format import format_bert_data_dataframe
from source.python.report.report_format import format_data_tune_dataframe
from source.python.report.report_format import format_cnn_tune_dataframe
from source.python.report.report_utils import convert_bert_step_to_epoch

def concat_tune_reports_format (reports : Dict, formatter : Callable, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	data = None

	if len(reports[mode]) == 0 :
		return None

	for key, dataframe in reports[mode].items() :
		keys = key.split('-')

		arch     = keys[0]
		sequence = keys[1]
		target0  = keys[5]
		target1  = keys[6] if len(keys) >= 7 else None
		target2  = keys[7] if len(keys) >= 8 else None

		dataframe = dataframe.copy()
		dataframe.insert(0, 'Model',    arch)
		dataframe.insert(1, 'Sequence', sequence)
		dataframe.insert(4, 'Target0',  target0)
		dataframe.insert(5, 'Target1',  target1)
		dataframe.insert(5, 'Target2',  target2)

		if data is None :
			data = dataframe
		else :
			data = pandas.concat((data, dataframe))

	if   mode == 'regression'     : sort_by = 'valid_r2'
	elif mode == 'classification' : sort_by = 'valid_accuracy'
	else                          : raise ValueError()

	data = data.sort_values(sort_by, ascending = False, na_position = 'last')
	data = data.reset_index()

	return formatter(
		dataframe = data,
		mode      = mode
	).head(n = n)

def concat_cnn_tune_reports (reports : Dict, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	return concat_tune_reports_format(
		reports   = reports,
		formatter = format_cnn_tune_dataframe,
		mode      = mode,
		n         = n
	)

def concat_data_tune_reports (reports : Dict, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	return concat_tune_reports_format(
		reports   = reports,
		formatter = format_data_tune_dataframe,
		mode      = mode,
		n         = n
	)

def concat_bert_reports (data : Dict[str, Any], mode : str, metric : str, ascending : bool) -> DataFrame :
	"""
	Doc
	"""

	array = []

	for key, dataframe in data[mode].items() :
		dataframe = dataframe.sort_values(metric, ascending = ascending)
		dataframe = dataframe.iloc[0, :]

		item = dataframe.to_dict()

		tokens = key.split('-')

		bert_arch     = tokens[0]
		bert_type     = tokens[1]
		bert_layer    = tokens[2]
		bert_kmer     = tokens[3]
		bert_sequence = tokens[4]
		bert_optim    = tokens[5]
		bert_epochs   = tokens[6]
		bert_target0  = tokens[7]
		bert_target1  = tokens[8] if len(tokens) >=  9 else None
		bert_target2  = tokens[9] if len(tokens) >= 10 else None

		item['Mode']      = str(mode)
		item['Arch']      = str(bert_arch)
		item['Type']      = str(bert_type)
		item['Layer']     = int(bert_layer)
		item['Kmer']      = int(bert_kmer)
		item['Sequence']  = str(bert_sequence)
		item['Optimizer'] = str(bert_optim)
		item['Epochs']    = int(bert_epochs)
		item['Target0']   = str(bert_target0)
		item['Target1']   = str(bert_target1)
		item['Target2']   = str(bert_target2)

		item['Epoch'] = convert_bert_step_to_epoch(
			step = item['step'],
			floor = True
		)

		array.append(item)

	return format_bert_data_dataframe(
		dataframe = DataFrame(array),
		mode      = mode
	)
