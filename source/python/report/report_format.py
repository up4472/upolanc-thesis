from pandas import DataFrame
from typing import Tuple
from typing import List

from source.python.report.report_constants import FLOAT_FORMAT

def format_cnn_tune_dataframe_metrics (dataframe : DataFrame, mode : str) -> Tuple[DataFrame, List] :
	"""
	Doc
	"""

	columns = list()

	if mode == 'regression' :
		columns = [
			'ID',
			'Valid_MSE', 'Valid_MAE', 'Valid_R2',
			'Train_MSE'
		]

		dataframe = dataframe.rename(columns = {
			'logdir'     : 'ID',
			'valid_loss' : 'Valid_MSE',
			'valid_mae'  : 'Valid_MAE',
			'valid_r2'   : 'Valid_R2',
			'train_loss' : 'Train_MSE'
		})

		dataframe = dataframe.astype({
			'ID'        : str,
			'Valid_MSE' : float,
			'Valid_MAE' : float,
			'Valid_R2'  : float,
			'Train_MSE' : float
		})

		dataframe['Valid_MSE'] = dataframe['Valid_MSE'].map(FLOAT_FORMAT.format)
		dataframe['Valid_MAE'] = dataframe['Valid_MAE'].map(FLOAT_FORMAT.format)
		dataframe['Valid_R2' ] = dataframe['Valid_R2' ].map(FLOAT_FORMAT.format)
		dataframe['Train_MSE'] = dataframe['Train_MSE'].map(FLOAT_FORMAT.format)

	if mode == 'classification' :
		columns = [
			'ID',
			'Valid_Entropy', 'Valid_Accuracy', 'Valid_F1', 'Valid_AUROC',
			'Train_Entropy'
		]

		dataframe = dataframe.rename(columns = {
			'logdir'         : 'ID',
			'valid_loss'     : 'Valid_Entropy',
			'valid_accuracy' : 'Valid_Accuracy',
			'valid_f1'       : 'Valid_F1',
			'valid_auroc'    : 'Valid_AUROC',
			'train_loss'     : 'Train_Entropy'
		})

		dataframe = dataframe.astype({
			'ID'             : str,
			'Valid_Entropy'  : float,
			'Valid_Accuracy' : float,
			'Valid_F1'       : float,
			'Valid_AUROC'    : float,
			'Train_Entropy'  : float
		})

		dataframe['Valid_Entropy' ] = dataframe['Valid_Entropy' ].map(FLOAT_FORMAT.format)
		dataframe['Valid_Accuracy'] = dataframe['Valid_Accuracy'].map(FLOAT_FORMAT.format)
		dataframe['Valid_F1'      ] = dataframe['Valid_F1'      ].map(FLOAT_FORMAT.format)
		dataframe['Valid_AUROC'   ] = dataframe['Valid_AUROC'   ].map(FLOAT_FORMAT.format)
		dataframe['Train_Entropy' ] = dataframe['Train_Entropy' ].map(FLOAT_FORMAT.format)

	return dataframe, columns

def format_cnn_tune_dataframe_params (dataframe : DataFrame) -> Tuple[DataFrame, List] :
	"""
	Doc
	"""

	columns = list()

	has_momentum = 'config/optimizer/momentum' in dataframe.columns
	has_beta     = 'config/optimizer/beta1' in dataframe.columns and 'config/optimizer/beta2' in dataframe.columns

	columns.extend(['Epoch', 'Optimizer', 'LR'])
	columns.extend(['Decay', 'Scheduler', 'Batch', 'Dropout'])

	if has_beta :
		dataframe = dataframe.rename(columns = {
			'config/optimizer/beta1' : 'Beta1',
			'config/optimizer/beta2' : 'Beta2'
		})

		dataframe['Beta1'] = dataframe['Beta1'].map(FLOAT_FORMAT.format)
		dataframe['Beta2'] = dataframe['Beta2'].map(FLOAT_FORMAT.format)

	if has_momentum :
		dataframe = dataframe.rename(columns = {
			'config/optimizer/momentum' : 'Momentum'
		})

		dataframe['Momentum'] = dataframe['Momentum'].map(FLOAT_FORMAT.format)

	dataframe = dataframe.rename(columns = {
		'training_iteration'        : 'Epoch',
		'config/optimizer/name'     : 'Optimizer',
		'config/optimizer/lr'       : 'LR',
		'config/optimizer/decay'    : 'Decay',
		'config/scheduler/name'     : 'Scheduler',
		'config/dataset/batch_size' : 'Batch',
		'config/model/dropout'      : 'Dropout'
	})

	dataframe = dataframe.astype({
		'Epoch'     : int,
		'Optimizer' : str,
		'LR'        : float,
		'Decay'     : float,
		'Scheduler' : str,
		'Batch'     : int,
		'Dropout'   : float
	})

	dataframe['LR'     ] = dataframe['LR'     ].map(FLOAT_FORMAT.format)
	dataframe['Decay'  ] = dataframe['Decay'  ].map(FLOAT_FORMAT.format)
	dataframe['Dropout'] = dataframe['Dropout'].map(FLOAT_FORMAT.format)

	return dataframe, columns

def format_cnn_tune_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = ['Model', 'Sequence', 'Filter', 'Target0', 'Target1', 'Target2']

	dataframe, columns_metric = format_cnn_tune_dataframe_metrics(dataframe = dataframe, mode = mode)
	dataframe, columns_params = format_cnn_tune_dataframe_params (dataframe = dataframe)

	columns.extend([item for item in columns_metric if item not in columns])
	columns.extend([item for item in columns_params if item not in columns])

	dataframe = dataframe[columns].copy()

	return dataframe

def format_data_tune_dataframe_metrics (dataframe : DataFrame, mode : str) -> Tuple[DataFrame, List] :
	"""
	Doc
	"""

	columns = list()

	if mode == 'regression' :
		columns = [
			'ID',
			'Valid_MSE', 'Valid_MAE', 'Valid_R2',
			'Train_MSE',
		]

		dataframe = dataframe.rename(columns = {
			'logdir'     : 'ID',
			'valid_loss' : 'Valid_MSE',
			'valid_mae'  : 'Valid_MAE',
			'valid_r2'   : 'Valid_R2',
			'train_loss' : 'Train_MSE'
		})

		dataframe = dataframe.astype({
			'ID'        : str,
			'Valid_MSE' : float,
			'Valid_MAE' : float,
			'Valid_R2'  : float,
			'Train_MSE' : float,
		})

		dataframe['Valid_MSE'] = dataframe['Valid_MSE'].map(FLOAT_FORMAT.format)
		dataframe['Valid_MAE'] = dataframe['Valid_MAE'].map(FLOAT_FORMAT.format)
		dataframe['Valid_R2' ] = dataframe['Valid_R2' ].map(FLOAT_FORMAT.format)
		dataframe['Train_MSE'] = dataframe['Train_MSE'].map(FLOAT_FORMAT.format)

	if mode == 'classification' :
		columns = [
			'ID',
			'Valid_Entropy', 'Valid_Accuracy', 'Valid_F1', 'Valid_AUROC',
			'Train_Entropy'
		]

		dataframe = dataframe.rename(columns = {
			'logdir'         : 'ID',
			'valid_loss'     : 'Valid_Entropy',
			'valid_accuracy' : 'Valid_Accuracy',
			'valid_f1'       : 'Valid_F1',
			'valid_auroc'    : 'Valid_AUROC',
			'train_loss'     : 'Train_Entropy'
		})

		dataframe = dataframe.astype({
			'ID'             : str,
			'Valid_Entropy'  : float,
			'Valid_Accuracy' : float,
			'Valid_F1'       : float,
			'Valid_AUROC'    : float,
			'Train_Entropy'  : float
		})

		dataframe['Valid_Entropy' ] = dataframe['Valid_Entropy' ].map(FLOAT_FORMAT.format)
		dataframe['Valid_Accuracy'] = dataframe['Valid_Accuracy'].map(FLOAT_FORMAT.format)
		dataframe['Valid_F1'      ] = dataframe['Valid_F1'      ].map(FLOAT_FORMAT.format)
		dataframe['Valid_AUROC'   ] = dataframe['Valid_AUROC'   ].map(FLOAT_FORMAT.format)
		dataframe['Train_Entropy' ] = dataframe['Train_Entropy' ].map(FLOAT_FORMAT.format)

	return dataframe, columns

def format_data_tune_dataframe_params (dataframe : DataFrame, mode : str) -> Tuple[DataFrame, List] :
	"""
	Doc
	"""

	columns = list()

	if mode == 'regression' :
		columns = [
			'Epoch', 'Lambda'
		]

		dataframe = dataframe.rename(columns = {
			'training_iteration'   : 'Epoch',
			'config/boxcox/lambda' : 'Lambda'
		})

		dataframe = dataframe.astype({
			'Epoch'  : int,
			'Lambda' : float,
		})

		dataframe['Lambda'] = dataframe['Lambda'].map(FLOAT_FORMAT.format)

	if mode == 'classification' :
		columns = [
			'Epoch', 'Lambda', 'Bins'
		]

		dataframe = dataframe.rename(columns = {
			'training_iteration'   : 'Epoch',
			'config/boxcox/lambda' : 'Lambda',
			'config/class/bins'    : 'Bins'
		})

		dataframe = dataframe.astype({
			'Epoch'  : int,
			'Lambda' : float,
			'Bins'   : int
		})

		dataframe['Lambda'] = dataframe['Lambda'].map(FLOAT_FORMAT.format)

	return dataframe, columns

def format_data_tune_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = ['Model', 'Sequence', 'Filter', 'Target0', 'Target1', 'Target2']

	dataframe, columns_metric = format_data_tune_dataframe_metrics(dataframe = dataframe, mode = mode)
	dataframe, columns_params = format_data_tune_dataframe_params (dataframe = dataframe, mode = mode)

	columns.extend([item for item in columns_metric if item not in columns])
	columns.extend([item for item in columns_params if item not in columns])

	dataframe = dataframe[columns].copy()

	return dataframe

def format_feature_tune_dataframe_metrics (dataframe : DataFrame, mode : str) -> Tuple[DataFrame, List] :
	"""
	Doc
	"""

	columns = list()

	if mode == 'regression' :
		columns = [
			'ID',
			'Valid_MSE', 'Valid_MAE', 'Valid_R2',
			'Train_MSE'
		]

		dataframe = dataframe.rename(columns = {
			'logdir'     : 'ID',
			'valid_loss' : 'Valid_MSE',
			'valid_mae'  : 'Valid_MAE',
			'valid_r2'   : 'Valid_R2',
			'train_loss' : 'Train_MSE'
		})

		dataframe = dataframe.astype({
			'ID'        : str,
			'Valid_MSE' : float,
			'Valid_MAE' : float,
			'Valid_R2'  : float,
			'Train_MSE' : float
		})

		dataframe['Valid_MSE'] = dataframe['Valid_MSE'].map(FLOAT_FORMAT.format)
		dataframe['Valid_MAE'] = dataframe['Valid_MAE'].map(FLOAT_FORMAT.format)
		dataframe['Valid_R2' ] = dataframe['Valid_R2' ].map(FLOAT_FORMAT.format)
		dataframe['Train_MSE'] = dataframe['Train_MSE'].map(FLOAT_FORMAT.format)

	return dataframe, columns

def format_feature_tune_dataframe_params (dataframe : DataFrame) -> Tuple[DataFrame, List] :
	"""
	Doc
	"""

	columns = [
		'Epoch', 'Optimizer', 'LR', 'Scheduler', 'Batch', 'Dropout', 'Features',
		'Filter', 'Model', 'Target'
	]

	dataframe = dataframe.rename(columns = {
		'training_iteration'        : 'Epoch',
		'config/optimizer/name'     : 'Optimizer',
		'config/optimizer/lr'       : 'LR',
		'config/optimizer/decay'    : 'Decay',
		'config/scheduler/name'     : 'Scheduler',
		'config/dataset/batch_size' : 'Batch',
		'config/model/dropout'      : 'Dropout',
		'config/model/fc1/features' : 'Features',
		'config/gs/filter'          : 'Filter',
		'config/gs/model'           : 'Model',
		'config/gs/target'          : 'Target'
	})

	dataframe['Filter'] = [
		item
		if 'filter' not in item
		else 'f' + item[6:]
		for item in dataframe['Filter']
	]

	dataframe = dataframe.astype({
		'Epoch'    : int,
		'LR'       : float,
		'Decay'    : float,
		'Batch'    : int,
		'Dropout'  : float,
		'Features' : int
	})

	dataframe['LR'     ] = dataframe['LR'     ].map(FLOAT_FORMAT.format)
	dataframe['Decay'  ] = dataframe['Decay'  ].map(FLOAT_FORMAT.format)
	dataframe['Dropout'] = dataframe['Dropout'].map(FLOAT_FORMAT.format)

	return dataframe, columns

def format_feature_tune_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = ['Model', 'Filter', 'Target0', 'Target1', 'Target2']

	dataframe, columns_metric = format_feature_tune_dataframe_metrics(dataframe = dataframe, mode = mode)
	dataframe, columns_params = format_feature_tune_dataframe_params (dataframe = dataframe)

	columns.extend([item for item in columns_metric if item not in columns])
	columns.extend([item for item in columns_params if item not in columns])

	dataframe = dataframe[columns].copy()

	return dataframe

def format_bert_data_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	if mode == 'regression' :
		dataframe = dataframe.rename(columns = {
			'eval_r2'        : 'Eval_R2',
			'eval_max_error' : 'Eval_ME',
			'eval_mape'      : 'Eval_MAPE',
			'eval_mae'       : 'Eval_MAE',
			'learning_rate'  : 'Learning_Rate',
			'loss'           : 'Train_MSE',
			'step'           : 'Step'
		})

		dataframe = dataframe.astype({
			'Layer'   : int,
			'Kmer'    : int,
			'Feature' : int,
			'Step'    : int,
			'Epoch'   : int
		})

		dataframe = dataframe[[
			'Mode', 'Arch', 'Type', 'Layer', 'Kmer', 'Feature', 'Filter', 'Sequence', 'Optimizer', 'Epochs',
			'Target0', 'Target1', 'Target2',
			'Eval_R2', 'Eval_ME', 'Eval_MAPE', 'Eval_MAE', 'Learning_Rate',
			'Step', 'Epoch'
		]]

		dataframe['Eval_R2'      ] = dataframe['Eval_R2'      ].map(FLOAT_FORMAT.format)
		dataframe['Eval_ME'      ] = dataframe['Eval_ME'      ].map(FLOAT_FORMAT.format)
		dataframe['Eval_MAPE'    ] = dataframe['Eval_MAPE'    ].map(FLOAT_FORMAT.format)
		dataframe['Eval_MAE'     ] = dataframe['Eval_MAE'     ].map(FLOAT_FORMAT.format)
		dataframe['Learning_Rate'] = dataframe['Learning_Rate'].map(FLOAT_FORMAT.format)

		dataframe = dataframe.sort_values('Eval_R2', ascending = False)

	return dataframe
