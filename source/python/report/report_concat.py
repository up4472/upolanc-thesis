from pandas import DataFrame
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import pandas

from source.python.report.report_format import format_bert_data_dataframe
from source.python.report.report_format import format_tune_data_dataframe
from source.python.report.report_format import format_tune_model_dataframe
from source.python.report.report_utils import convert_bert_step_to_epoch

def concat_tune_cnn_reports (reports : Dict, formatter : Callable, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	data = None

	if len(reports[mode]) == 0 :
		return None

	for key, dataframe in reports[mode].items() :
		keys = key.split('-')

		model    = keys[3]
		sequence = keys[4]
		target0  = keys[7]
		target1  = keys[8]
		target2  = None

		if len(keys) == 10 :
			target2 = keys[9]

		dataframe = dataframe.copy()
		dataframe.insert(0, 'Model', model)
		dataframe.insert(1, 'Sequence', sequence)
		dataframe.insert(4, 'Target0', target0)
		dataframe.insert(5, 'Target1', target1)
		dataframe.insert(5, 'Target2', target2)

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

def concat_tune_model_reports (reports : Dict, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	return concat_tune_cnn_reports(
		reports   = reports,
		formatter = format_tune_model_dataframe,
		mode      = mode,
		n         = n
	)

def concat_tune_data_reports (reports : Dict, mode : str, n : int = 50) -> Optional[DataFrame] :
	"""
	Doc
	"""

	return concat_tune_cnn_reports(
		reports   = reports,
		formatter = format_tune_data_dataframe,
		mode      = mode,
		n         = n
	)

def concat_bert_best (data : Dict[str, Any], mode : str, metric : str, ascending : bool) -> DataFrame :
	"""
	Doc
	"""

	array = []

	for key, dataframe in data[mode].items() :
		dataframe = dataframe.sort_values(metric, ascending = ascending)
		dataframe = dataframe.iloc[0, :]

		item = dataframe.to_dict()

		tokens = key.split('-')

		item['Mode']      = str(tokens[1])
		item['Model']     = '{}-{}'.format(tokens[2], tokens[3])
		item['Freeze']    = int(tokens[4])
		item['Kmer']      = int(tokens[5])
		item['Sequence']  = str(tokens[6])
		item['Optimizer'] = str(tokens[7])
		item['Epochs']    = int(tokens[8])
		item['Target0']   = str(tokens[9])
		item['Target1']   = str(tokens[10])
		item['Target2']   = str(tokens[11]) if len(tokens) == 12 else None
		item['Epoch']     = convert_bert_step_to_epoch(
			step  = item['step'],
			floor = True
		)

		array.append(item)

	return format_bert_data_dataframe(
		dataframe = DataFrame(array),
		mode      = mode
	)
