from pandas import DataFrame
from typing import Dict

from IPython.display import display

import itertools
import pandas
import os

from source.python.report.report_constants import BERT_ARCH
from source.python.report.report_constants import BERT_LAYERS
from source.python.report.report_constants import BERT_OUTPUT
from source.python.report.report_constants import BERT_POOLER
from source.python.report.report_constants import CNNS
from source.python.report.report_constants import EPOCHS
from source.python.report.report_constants import FCS
from source.python.report.report_constants import FEATURES
from source.python.report.report_constants import FILTERS
from source.python.report.report_constants import FLOAT_FORMAT
from source.python.report.report_constants import KMERS
from source.python.report.report_constants import OPTIMIZERS
from source.python.report.report_constants import PARAMS
from source.python.report.report_constants import SEQUENCES
from source.python.report.report_constants import TARGETS
from source.python.report.report_constants import TRIALS
from source.python.io.loader               import load_json
from source.python.report.report_format    import format_cnn_tune_dataframe
from source.python.report.report_format    import format_data_tune_dataframe
from source.python.report.report_utils     import convert_errlog_to_dataframe
from source.python.report.report_utils     import convert_json_to_dataframe
from source.python.report.report_utils     import recover_dataframe

#
# Folders
#

CNN_MODEL     = '{}-cnn'
DENSE_MODEL   = '{}-dense'
CNN_BERT      = '{}-cnn-bert'
TUNER_CNN     = '{}-tuner-cnn'
TUNER_DATA    = '{}-tuner-data'
TUNER_FEATURE = '{}-tuner-feature'
BERT_MODEL    = '{}-bert'

#
# CNN
#

def load_cnn_reports_for (root : str, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = [
		'Model', 'Param', 'Sequence', 'Filter', 'Epochs', 'Features', 'Target0', 'Target1', 'Target2',
		'Optimizer', 'LR', 'Beta1', 'Beta2', 'Decay', 'Dropout', 'Scheduler', 'Gamma',
		'Batch', 'Epoch'
	]

	if mode == 'regression'     : columns.extend(['MSE', 'R2'])
	if mode == 'classification' : columns.extend(['Entropy', 'Accuracy'])

	dataframe = DataFrame(columns = columns)
	root      = os.path.join(root, CNN_MODEL.format(mode))

	for items in itertools.product(CNNS, PARAMS, SEQUENCES, FILTERS, EPOCHS, FEATURES, TARGETS) :
		key    = '{:s}-{:s}-{:s}-{:s}-{:04d}-{:02d}-{:s}'.format(*items)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		config = os.path.join(folder, 'config.json')
		report = os.path.join(folder, 'report.json')

		if not os.path.exists(config) : continue
		if not os.path.exists(report) : continue

		config = load_json(filename = config)
		report = load_json(filename = report)

		target = items[-1].split('-')

		data = [
			str(items[0]),
			int(items[1]),
			str(items[2]),
			str(items[3]),
			int(items[4]),
			int(items[5]),
			target[0] if len(target) >= 1 else None,
			target[1] if len(target) >= 2 else None,
			target[2] if len(target) >= 3 else None
		]

		data.extend([
			config['optimizer/name'],
			config['optimizer/lr'],
			config['optimizer/beta1'],
			config['optimizer/beta2'],
			config['optimizer/decay'],
			config['model/dropout'],
			config['scheduler/name'],
			config['scheduler/exponential/factor'],
			config['dataset/batch/train'],
			report['evaluation/best/epoch'],
			report['evaluation/best/loss']
		])

		if mode == 'regression' :
			data.extend([report['evaluation/best/test/r2/mean']])
		elif mode == 'classification' :
			data.extend([report['evaluation/best/test/accuracy/mean']])

		dataframe.loc[-1] = data
		dataframe.index   = dataframe.index + 1
		dataframe         = dataframe.sort_index()

	return dataframe

def load_cnn_reports (root : str) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	reports = {
		'regression'     : load_cnn_reports_for(root = root, mode = 'regression'),
		'classification' : load_cnn_reports_for(root = root, mode = 'classification')
	}

	for mode in ['regression', 'classification'] :
		reports[mode]['LR'     ] = reports[mode]['LR'     ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Beta1'  ] = reports[mode]['Beta1'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Beta2'  ] = reports[mode]['Beta2'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Decay'  ] = reports[mode]['Decay'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Gamma'  ] = reports[mode]['Gamma'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Dropout'] = reports[mode]['Dropout'].astype(float).map(FLOAT_FORMAT.format)

	reports['regression']['MSE'] = reports['regression']['MSE'].astype(float).map(FLOAT_FORMAT.format)
	reports['regression']['R2' ] = reports['regression']['R2' ].astype(float).map(FLOAT_FORMAT.format)

	reports['regression'] = reports['regression'].sort_values('R2', ascending = False)

	reports['classification']['Entropy' ] = reports['classification']['Entropy' ].astype(float).map(FLOAT_FORMAT.format)
	reports['classification']['Accuracy'] = reports['classification']['Accuracy'].astype(float).map(FLOAT_FORMAT.format)

	reports['classification'] = reports['classification'].sort_values('Accuracy', ascending = False)

	return reports

def load_fc_reports_for (root : str, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = [
		'Arch', 'Model', 'Param', 'Filter', 'Epochs', 'Target0', 'Target1', 'Target2',
		'FC1', 'FC2', 'Optimizer', 'LR', 'Beta1', 'Beta2', 'Decay', 'Dropout', 'Scheduler', 'Gamma',
		'Batch', 'Epoch'
	]

	if mode == 'regression'     : columns.extend(['MSE', 'R2'])
	if mode == 'classification' : columns.extend(['Entropy', 'Accuracy'])

	dataframe = DataFrame(columns = columns)
	root      = os.path.join(root, DENSE_MODEL.format(mode))

	for items in itertools.product(FCS, CNNS, PARAMS, FILTERS, EPOCHS, TARGETS) :
		key    = '{:s}-{:s}-{:s}-{:s}-{:04d}-{:s}'.format(*items)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		config = os.path.join(folder, 'config.json')
		report = os.path.join(folder, 'report.json')

		if not os.path.exists(config) : continue
		if not os.path.exists(report) : continue

		config = load_json(filename = config)
		report = load_json(filename = report)

		target = items[-1].split('-')

		data = [
			str(items[0]),
			str(items[1]),
			int(items[2]),
			str(items[3]),
			int(items[4]),
			target[0] if len(target) >= 1 else None,
			target[1] if len(target) >= 2 else None,
			target[2] if len(target) >= 3 else None
		]

		data.extend([
			config['model/fc1/features'],
			config['model/fc2/features'],
			config['optimizer/name'],
			config['optimizer/lr'],
			config['optimizer/beta1'],
			config['optimizer/beta2'],
			config['optimizer/decay'],
			config['model/dropout'],
			config['scheduler/name'],
			config['scheduler/exponential/factor'],
			config['dataset/batch/train'],
			report['evaluation/best/epoch'],
			report['evaluation/best/loss']
		])

		if mode == 'regression' :
			data.extend([report['evaluation/best/test/r2/mean']])
		elif mode == 'classification' :
			data.extend([report['evaluation/best/test/accuracy/mean']])

		dataframe.loc[-1] = data
		dataframe.index   = dataframe.index + 1
		dataframe         = dataframe.sort_index()

	return dataframe

def load_fc_reports (root : str) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	reports = {
		'regression'     : load_fc_reports_for(root = root, mode = 'regression'),
		'classification' : load_fc_reports_for(root = root, mode = 'classification')
	}

	for mode in ['regression', 'classification'] :
		reports[mode]['LR'     ] = reports[mode]['LR'     ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Beta1'  ] = reports[mode]['Beta1'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Beta2'  ] = reports[mode]['Beta2'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Decay'  ] = reports[mode]['Decay'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Gamma'  ] = reports[mode]['Gamma'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Dropout'] = reports[mode]['Dropout'].astype(float).map(FLOAT_FORMAT.format)

	reports['regression']['MSE'] = reports['regression']['MSE'].astype(float).map(FLOAT_FORMAT.format)
	reports['regression']['R2' ] = reports['regression']['R2' ].astype(float).map(FLOAT_FORMAT.format)

	reports['regression'] = reports['regression'].sort_values('R2', ascending = False)

	reports['classification']['Entropy' ] = reports['classification']['Entropy' ].astype(float).map(FLOAT_FORMAT.format)
	reports['classification']['Accuracy'] = reports['classification']['Accuracy'].astype(float).map(FLOAT_FORMAT.format)

	reports['classification'] = reports['classification'].sort_values('Accuracy', ascending = False)

	return reports

def load_bert_cnn_reports_for (root : str, mode : str = 'regression') -> DataFrame :
	"""
	Doc
	"""

	columns = [
		'Model', 'Sequence', 'Filter', 'KMer', 'Features', 'Target0', 'Target1', 'Target2',
		'Optimizer', 'LR', 'Beta1', 'Beta2', 'Decay', 'Dropout', 'Scheduler', 'Gamma',
		'Batch', 'Epoch'
	]

	if mode == 'regression'     : columns.extend(['MSE', 'R2'])
	if mode == 'classification' : columns.extend(['Entropy', 'Accuracy'])

	dataframe = DataFrame(columns = columns)
	root      = os.path.join(root, CNN_BERT.format(mode))

	for items in itertools.product(BERT_ARCH, BERT_OUTPUT, KMERS, FILTERS, SEQUENCES, EPOCHS, FEATURES, TARGETS) :
		key    = '{:s}-{:s}-{:d}-{:s}-{:s}-{:04d}-{:02d}-{:s}'.format(*items)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		config = os.path.join(folder, 'config.json')
		report = os.path.join(folder, 'report.json')

		if not os.path.exists(config) : continue
		if not os.path.exists(report) : continue

		config = load_json(filename = config)
		report = load_json(filename = report)

		target = items[-1].split('-')

		data = [
			str(items[0]) + '-' + str(items[1]),
			str(items[4]),
			str(items[3]),
			int(items[2]),
			int(items[6]),
			target[0] if len(target) >= 1 else None,
			target[1] if len(target) >= 2 else None,
			target[2] if len(target) >= 3 else None
		]

		data.extend([
			config['optimizer/name'],
			config['optimizer/lr'],
			config['optimizer/beta1'],
			config['optimizer/beta2'],
			config['optimizer/decay'],
			config['model/dropout'],
			config['scheduler/name'],
			config['scheduler/exponential/factor'],
			config['dataset/batch/train'],
			report['evaluation/best/epoch'],
			report['evaluation/best/loss']
		])

		if mode == 'regression' :
			data.extend([report['evaluation/best/test/r2/mean']])
		elif mode == 'classification' :
			data.extend([report['evaluation/best/test/accuracy/mean']])

		dataframe.loc[-1] = data
		dataframe.index   = dataframe.index + 1
		dataframe         = dataframe.sort_index()

	return dataframe

def load_bert_cnn_reports (root : str) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	reports = {
		'regression'     : load_bert_cnn_reports_for(root = root, mode = 'regression'),
		'classification' : load_bert_cnn_reports_for(root = root, mode = 'classification')
	}

	for mode in ['regression', 'classification'] :
		reports[mode]['LR'     ] = reports[mode]['LR'     ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Beta1'  ] = reports[mode]['Beta1'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Beta2'  ] = reports[mode]['Beta2'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Decay'  ] = reports[mode]['Decay'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Gamma'  ] = reports[mode]['Gamma'  ].astype(float).map(FLOAT_FORMAT.format)
		reports[mode]['Dropout'] = reports[mode]['Dropout'].astype(float).map(FLOAT_FORMAT.format)

	reports['regression']['MSE'] = reports['regression']['MSE'].astype(float).map(FLOAT_FORMAT.format)
	reports['regression']['R2' ] = reports['regression']['R2' ].astype(float).map(FLOAT_FORMAT.format)

	reports['regression'] = reports['regression'].sort_values('R2', ascending = False)

	reports['classification']['Entropy' ] = reports['classification']['Entropy' ].astype(float).map(FLOAT_FORMAT.format)
	reports['classification']['Accuracy'] = reports['classification']['Accuracy'].astype(float).map(FLOAT_FORMAT.format)

	reports['classification'] = reports['classification'].sort_values('Accuracy', ascending = False)

	return reports

#
# Tuner
#

def load_cnn_tune_reports_for (root : str, mode : str, n : int = 5, show : bool = False) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	report = dict()
	root   = os.path.join(root, TUNER_CNN.format(mode))

	for items in itertools.product(CNNS, SEQUENCES, FILTERS, TRIALS, EPOCHS, FEATURES, TARGETS) :
		key    = '{:s}-{:s}-{:s}-{:04d}-{:2d}-{:02d}-{:s}'.format(*items)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		filename = os.path.join(folder, 'report.csv')
		logdir   = os.path.join(folder, 'raytune')

		if not os.path.exists(filename) :
			dataframe = recover_dataframe(logdir)
			dataframe.to_csv(filename)
		else :
			dataframe = pandas.read_csv(filename, index_col = [0])
			dataframe = dataframe.sort_values('valid_loss', ascending = True)
			dataframe = dataframe.reset_index()

		dataframe['logdir'] = dataframe['logdir'].str.split('/')
		dataframe['logdir'] = dataframe['logdir'].str[-1]

		print(filename)

		if show :
			display(format_cnn_tune_dataframe(
				dataframe = dataframe,
				mode      = mode
			).head(n = n))

		report[key] = dataframe

	print()

	return report

def load_cnn_tune_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'regression'     : load_cnn_tune_reports_for(root = root, n = n, show = show, mode = 'regression'),
		'classification' : load_cnn_tune_reports_for(root = root, n = n, show = show, mode = 'classification')
	}


def load_data_tune_reports_for (root : str, mode : str, n : int = 5, show : bool = False) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	report = dict()
	root   = os.path.join(root, TUNER_DATA.format(mode))

	for items in itertools.product(CNNS, SEQUENCES, FILTERS, TRIALS, EPOCHS, TARGETS) :
		key    = '{:s}-{:s}-{:s}-{:04d}-{:2d}-{:s}'.format(*items)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		filename = os.path.join(folder, 'report.csv')
		logdir   = os.path.join(folder, 'raytune')

		if not os.path.exists(filename) :
			dataframe = recover_dataframe(logdir)
			dataframe.to_csv(filename)
		else :
			dataframe = pandas.read_csv(filename, index_col = [0])
			dataframe = dataframe.sort_values('valid_loss', ascending = True)
			dataframe = dataframe.reset_index()

		dataframe['logdir'] = dataframe['logdir'].str.split('/')
		dataframe['logdir'] = dataframe['logdir'].str[-1]

		print(filename)

		if show :
			display(format_data_tune_dataframe(
				dataframe = dataframe,
				mode      = mode
			).head(n = n))

		report[key] = dataframe

	print()

	return report

def load_data_tune_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'regression'     : load_data_tune_reports_for(root = root, n = n, show = show, mode = 'regression'),
		'classification' : load_data_tune_reports_for(root = root, n = n, show = show, mode = 'classification')
	}

def load_feature_tune_reports_for (root : str, mode : str, n : int = 5, show : bool = False) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	report = dict()
	root   = os.path.join(root, TUNER_FEATURE.format(mode))

	for items in itertools.product(FEATURES, TRIALS, EPOCHS) :
		key    = '{:2d}-{:04d}-{:2d}'.format(*items)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		filename = os.path.join(folder, 'report.csv')
		logdir   = os.path.join(folder, 'raytune')

		if not os.path.exists(filename) :
			dataframe = recover_dataframe(logdir)
			dataframe.to_csv(filename)
		else :
			dataframe = pandas.read_csv(filename, index_col = [0])
			dataframe = dataframe.sort_values('valid_loss', ascending = True)
			dataframe = dataframe.reset_index()

		dataframe['logdir'] = dataframe['logdir'].str.split('/')
		dataframe['logdir'] = dataframe['logdir'].str[-1]

		print(filename)

		if show :
			display(format_data_tune_dataframe(
				dataframe = dataframe,
				mode      = mode
			).head(n = n))

		report[key] = dataframe

	print()

	return report

def load_feature_tune_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'regression'     : load_feature_tune_reports_for(root = root, n = n, show = show, mode = 'regression'),
		'classification' : load_feature_tune_reports_for(root = root, n = n, show = show, mode = 'classification')
	}

#
# Bert
#

def load_bert_reports_for (root : str, mode : str, n : int = 5, show : bool = False) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	report = dict()
	root   = os.path.join(root, BERT_MODEL.format(mode))

	for config in itertools.product(BERT_OUTPUT, BERT_ARCH, BERT_POOLER, BERT_LAYERS, KMERS, FEATURES, SEQUENCES, OPTIMIZERS, FILTERS, EPOCHS, TARGETS) :
		key    = '{:s}-{:s}-{:s}-{:02d}-{:d}-{:02d}-{:s}-{:s}-{:s}-{:04d}-{:s}'.format(*config)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		file_json = os.path.join(folder, 'results.json')
		file_json = os.path.exists(file_json)

		if not file_json :
			errlogs = [file for file in os.listdir(folder) if file.endswith('.err')]

			dataframe = convert_errlog_to_dataframe(
				root        = folder,
				source_name = errlogs[0],
				target_name = 'results.csv',
				step_size   = 100
			)
		else :
			dataframe = convert_json_to_dataframe(
				root        = folder,
				source_name = 'results.json',
				target_name = 'results.csv'
			)

		print(folder)

		if show :
			display(format_cnn_tune_dataframe(
				dataframe = dataframe,
				mode      = mode
			).head(n = n))

		report[key] = dataframe

	print()

	return report

def load_bert_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'regression'     : load_bert_reports_for(root = root, n = n, show = show, mode = 'regression'),
		'classification' : load_bert_reports_for(root = root, n = n, show = show, mode = 'classification')
	}
