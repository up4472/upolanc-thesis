from pandas import DataFrame
from typing import Dict

from IPython.display import display

import itertools
import pandas
import os

from source.python.io.loader            import load_json
from source.python.report.report_format import format_tune_data_dataframe
from source.python.report.report_format import format_tune_model_dataframe
from source.python.report.report_utils  import convert_json_to_dataframe
from source.python.report.report_utils  import recover_dataframe

MODES        = ['r', 'c']
CNNS         = ['zrimec', 'washburn']
TRANSFORMERS = ['bertfc3-rnn', 'bertfc3-cat', 'bertfc3-def']
OPTIMIZERS   = ['adam', 'lamb']
FREEZED      = [9, 12]
SEQUENCES    = ['gene', 'transcript', 'promoter']
KMERS        = [3, 4, 5, 6]
EPOCHS       = [1000, 500, 250, 200, 150, 100, 25]
TRIALS       = [1000, 500, 250, 200, 150, 100, 25]
TARGETS      = [
	['tissue', 'group', 'global'],
	['mean', 'max'],
	['all', 'explode', 'seedling', None]
]

def load_tune_model_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	reports = {
		'regression'     : dict(),
		'classification' : dict()
	}

	for item in itertools.product(MODES, CNNS, SEQUENCES, TRIALS, EPOCHS, TARGETS[0], TARGETS[1], TARGETS[2]) :
		mode    = item[0]
		model   = item[1]
		data    = item[2]
		trial   = item[3]
		epoch   = item[4]
		target0 = item[5]
		target1 = item[6]
		target2 = item[7]

		if target2 is None :
			template = 'tune-model-{}-{}-{}-{:04d}-{}-{}-{}'
			items    = [mode, model, data, trial, epoch, target0, target1]
		else :
			template = 'tune-model-{}-{}-{}-{:04d}-{}-{}-{}-{}'
			items    = [mode, model, data, trial, epoch, target0, target1, target2]

		key      = template.format(*items)
		folder   = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		if   mode[0] == 'r' : mode = 'regression'
		elif mode[0] == 'c' : mode = 'classification'

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
			display(format_tune_model_dataframe(
				dataframe = dataframe,
				mode      = mode
			).head(n = n))

		reports[mode][key] = dataframe

	return reports

def load_tune_data_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	reports = {
		'regression'     : dict(),
		'classification' : dict()
	}

	for item in itertools.product(MODES, CNNS, SEQUENCES, TRIALS, EPOCHS, TARGETS[0], TARGETS[1], TARGETS[2]) :
		mode    = item[0]
		model   = item[1]
		data    = item[2]
		trial   = item[3]
		epoch   = item[4]
		target0 = item[5]
		target1 = item[6]
		target2 = item[7]

		if target2 is None :
			template = 'tune-data-{}-{}-{}-{:04d}-{}-{}-{}'
			items    = [mode, model, data, trial, epoch, target0, target1]
		else :
			template = 'tune-data-{}-{}-{}-{:04d}-{}-{}-{}-{}'
			items    = [mode, model, data, trial, epoch, target0, target1, target2]

		key      = template.format(*items)
		folder   = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		if   mode[0] == 'r' : mode = 'regression'
		elif mode[0] == 'c' : mode = 'classification'

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
			display(format_tune_data_dataframe(
				dataframe = dataframe,
				mode      = mode
			).head(n = n))

		reports[mode][key] = dataframe

	return reports

def load_cnn_reports (root : str) -> Dict :
	"""
	Doc
	"""

	reg_df = DataFrame(columns = [
		'Type', 'Epochs', 'Target_0', 'Target_1', 'Target_2', 'Model',
		'Optimizer', 'Learning_Rate', 'Momentum', 'Decay', 'Scheduler',
		'Batch_Size', 'Dropout', 'Epoch',
		'Valid_MSE', 'Eval_MSE', 'Eval_MAE', 'Eval_R2'
	])

	cls_df = DataFrame(columns = [
		'Type', 'Epochs', 'Target_0', 'Target_1', 'Target_2', 'Model',
		'Optimizer', 'Learning_Rate', 'Momentum', 'Decay', 'Scheduler',
		'Batch_Size', 'Dropout', 'Epoch',
		'Valid_Entropy', 'Eval_Entropy', 'Eval_Accuracy', 'Eval_F1', 'Eval_AUROC'
	])

	reports = {
		'regression'     : reg_df,
		'classification' : cls_df
	}

	for item in itertools.product(MODES, CNNS, SEQUENCES, EPOCHS, TARGETS[0], TARGETS[1], TARGETS[2]) :
		mode    = item[0]
		model   = item[1]
		data    = item[2]
		epoch   = item[3]
		target0 = item[4]
		target1 = item[5]
		target2 = item[6]

		if target2 is None :
			template = 'model-{}-{}-{}-{:04d}-{}-{}'
			items    = [mode, model, data, epoch, target0, target1]
		else :
			template = 'model-{}-{}-{}-{:04d}-{}-{}-{}'
			items    = [mode, model, data, epoch, target0, target1, target2]

		key    = template.format(*items)
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

		data = [
			data, epoch, target0, target1, target2, model,
			config['optimizer/name'],
			config['optimizer/lr'],
			config['optimizer/momentum'],
			config['optimizer/decay'],
			config['scheduler/name'],
			config['dataset/batch/train'],
			config['model/dropout'],
			report['evaluation/best/epoch'],
			report['evaluation/best/loss']
		]

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

		reports[mode].loc[-1] = data
		reports[mode].index   = reports[mode].index + 1
		reports[mode]         = reports[mode].sort_index()

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

def load_bert_reports (root : str, n : int = 5, show : bool = False) -> Dict[str, Dict] :
	"""
	Doc
	"""

	reports = {
		'regression'     : dict(),
		'classification' : dict()
	}

	for item in itertools.product(MODES, TRANSFORMERS, FREEZED, KMERS, SEQUENCES, OPTIMIZERS, EPOCHS, TARGETS[0], TARGETS[1], TARGETS[2]) :
		mode    = item[0]
		model   = item[1]
		freeze  = item[2]
		kmer    = item[3]
		data    = item[4]
		optim   = item[5]
		epoch   = item[6]
		target0 = item[7]
		target1 = item[8]
		target2 = item[9]

		if target2 is None :
			template = 'model-{}-{}-{:02d}-{}-{}-{}-{:04d}-{}-{}'
			items    = [mode, model, freeze, kmer, data, optim, epoch, target0, target1]
		else :
			template = 'model-{}-{}-{:02d}-{}-{}-{}-{:04d}-{}-{}-{}'
			items    = [mode, model, freeze, kmer, data, optim, epoch, target0, target1, target2]

		key    = template.format(*items)
		folder = os.path.join(root, key)

		if not os.path.exists(folder) :
			continue

		if   mode[0] == 'r' : mode = 'regression'
		elif mode[0] == 'c' : mode = 'classification'

		dataframe = convert_json_to_dataframe(
			root        = folder,
			source_name = 'results.json',
			target_name = 'results.csv'
		)

		print(folder)

		if show :
			display(format_tune_model_dataframe(
				dataframe = dataframe,
				mode      = mode
			).head(n = n))

		reports[mode][key] = dataframe

	return reports
