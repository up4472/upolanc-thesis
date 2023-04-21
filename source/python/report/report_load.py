from pandas import DataFrame
from typing import Dict

from IPython.display import display

import itertools
import pandas
import os

from source.python.io.loader            import load_json
from source.python.report.report_format import format_cnn_tune_dataframe
from source.python.report.report_format import format_data_tune_dataframe
from source.python.report.report_utils  import convert_json_to_dataframe
from source.python.report.report_utils  import recover_dataframe

TARGETS = [
	'global-mean',
	'tissue-mean',
	'tissue-mean-seedling',
	'tissue-mean-explode'
]

def load_cnn_tune_reports_for (root : str, mode : str, n : int = 5, show : bool = False) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	report = dict()

	if mode == 'regression'     : root = os.path.join(root, 'cnn-tune-regression')
	if mode == 'classification' : root = os.path.join(root, 'cnn-tune-classification')

	cnn_archs      = ['zrimec', 'washburn']
	cnn_sequences  = ['promoter', 'transcript']
	cnn_trials     = [250, 500, 1000]
	cnn_epochs     = [25, 50]
	cnn_targets    = TARGETS

	for items in itertools.product(cnn_archs, cnn_sequences, cnn_trials, cnn_epochs, cnn_targets) :
		key    = '{:s}-{:s}-{:04d}-{:2d}-{:s}'.format(*items)
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

	if mode == 'regression'     : root = os.path.join(root, 'data-tune-regression')
	if mode == 'classification' : root = os.path.join(root, 'data-tune-classification')

	cnn_archs      = ['zrimec', 'washburn']
	cnn_sequences  = ['promoter', 'transcript']
	cnn_trials     = [250, 500, 1000]
	cnn_epochs     = [25, 50]
	cnn_targets    = TARGETS

	for items in itertools.product(cnn_archs, cnn_sequences, cnn_trials, cnn_epochs, cnn_targets) :
		key    = '{:s}-{:s}-{:04d}-{:2d}-{:s}'.format(*items)
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

	return report

def load_data_tune_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'regression'     : load_data_tune_reports_for(root = root, n = n, show = show, mode = 'regression'),
		'classification' : load_data_tune_reports_for(root = root, n = n, show = show, mode = 'classification')
	}

def load_cnn_reports_for (root : str, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = [
		'Model', 'Type', 'Epochs', 'Target_0', 'Target_1', 'Target_2',
		'Optimizer', 'Learning_Rate', 'Momentum', 'Decay', 'Scheduler',
		'Batch_Size', 'Dropout', 'Epoch'
	]

	if mode == 'regression'     : columns.extend(['Valid_MSE', 'Eval_MSE', 'Eval_MAE', 'Eval_R2'])
	if mode == 'classification' : columns.extend(['Valid_Entropy', 'Eval_Entropy', 'Eval_Accuracy', 'Eval_F1', 'Eval_AUROC'])

	dataframe = DataFrame(columns = columns)

	if mode == 'regression'     : root = os.path.join(root, 'cnn-regression')
	if mode == 'classification' : root = os.path.join(root, 'cnn-classification')

	cnn_archs      = ['zrimec', 'washburn']
	cnn_sequences  = ['promoter', 'transcript']
	cnn_epochs     = [250, 500, 1000]
	cnn_targets    = [
		'global-mean',
		'tissue-mean',
		'tissue-mean-seedling',
		'tissue-mean-explode'
	]

	for items in itertools.product(cnn_archs, cnn_sequences, cnn_epochs, cnn_targets) :
			key    = '{:s}-{:s}-{:04d}-{:s}'.format(*items)
			folder = os.path.join(root, key)

			if not os.path.exists(folder) :
				continue

			if   mode[0] == 'r' : mode = 'regression'
			elif mode[0] == 'c' : mode = 'classification'

			config = os.path.join(folder, 'config.json')
			report = os.path.join(folder, 'report.json')

			if not os.path.exists(config) : continue
			if not os.path.exists(report) : continue

			config = load_json(filename = config)
			report = load_json(filename = report)

			target = items[-1].split('-')

			data = [
				items[0],
				items[1],
				items[2],
				target[0] if len(target) >= 1 else None,
				target[1] if len(target) >= 2 else None,
				target[2] if len(target) >= 3 else None
			]

			data.extend([
				config['optimizer/name'],
				config['optimizer/lr'],
				config['optimizer/momentum'],
				config['optimizer/decay'],
				config['scheduler/name'],
				config['dataset/batch/train'],
				config['model/dropout'],
				report['evaluation/best/epoch'],
				report['evaluation/best/loss']
			])

			if mode == 'regression' :
				data.extend([
					report['evaluation/best/mse/mean'],
					report['evaluation/best/mae/mean'],
					report['evaluation/best/r2/mean']
				])
			elif mode == 'classification' :
				data.extend([
					report['evaluation/best/accuracy/mean'],
					report['evaluation/best/f1/mean'],
					report['evaluation/best/auroc/mean']
				])

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
		reports[mode]['Learning_Rate'] = reports[mode]['Learning_Rate'].astype(float).map('{:.9f}'.format)
		reports[mode]['Momentum'     ] = reports[mode]['Momentum'     ].astype(float).map('{:.9f}'.format)
		reports[mode]['Decay'        ] = reports[mode]['Decay'        ].astype(float).map('{:.9f}'.format)
		reports[mode]['Dropout'      ] = reports[mode]['Dropout'      ].astype(float).map('{:.3f}'.format)

	reports['regression']['Valid_MSE'] = reports['regression']['Valid_MSE'].astype(float).map('{:.9f}'.format)
	reports['regression']['Eval_MSE' ] = reports['regression']['Eval_MSE' ].astype(float).map('{:.9f}'.format)
	reports['regression']['Eval_MAE' ] = reports['regression']['Eval_MAE' ].astype(float).map('{:.9f}'.format)
	reports['regression']['Eval_R2'  ] = reports['regression']['Eval_R2'  ].astype(float).map('{:.9f}'.format)

	reports['regression'] = reports['regression'].sort_values('Eval_R2', ascending = False)

	reports['classification']['Valid_Entropy'] = reports['classification']['Valid_Entropy'].astype(float).map('{:.9f}'.format)
	reports['classification']['Eval_Entropy' ] = reports['classification']['Eval_Entropy' ].astype(float).map('{:.9f}'.format)
	reports['classification']['Eval_Accuracy'] = reports['classification']['Eval_Accuracy'].astype(float).map('{:.9f}'.format)
	reports['classification']['Eval_F1'      ] = reports['classification']['Eval_F1'      ].astype(float).map('{:.9f}'.format)
	reports['classification']['Eval_AUROC'   ] = reports['classification']['Eval_AUROC'   ].astype(float).map('{:.9f}'.format)

	reports['classification'] = reports['classification'].sort_values('Eval_Accuracy', ascending = False)

	return reports

def load_bert_reports_for (root : str, mode : str, n : int = 5, show : bool = False) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	report = dict()

	if mode == 'regression'     : root = os.path.join(root, 'dnabert-regression')
	if mode == 'classification' : root = os.path.join(root, 'dnabert-classification')

	bert_archs      = ['fc2', 'fc3']
	bert_types      = ['def', 'rnn', 'cat']
	bert_layers     = [9, 11, 12]
	bert_kmers      = [3, 6]
	bert_features   = [0, 72, 77]
	bert_sequences  = ['promoter', 'transcript']
	bert_optimizers = ['adam', 'lamb']
	bert_epochs     = [150, 250]
	bert_targets    = TARGETS

	for config in itertools.product(bert_archs, bert_types, bert_layers, bert_kmers, bert_features, bert_sequences, bert_optimizers, bert_epochs, bert_targets) :
		key    = '{:s}-{:s}-{:02d}-{:d}-{:02d}-{:s}-{:s}-{:04d}-{:s}'.format(*config)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

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

	return report

def load_bert_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'regression'     : load_bert_reports_for(root = root, n = n, show = show, mode = 'regression'),
		'classification' : load_bert_reports_for(root = root, n = n, show = show, mode = 'classification')
	}
