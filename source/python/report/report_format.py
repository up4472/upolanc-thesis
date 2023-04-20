from pandas import DataFrame
from typing import Tuple
from typing import List

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

		dataframe['Valid_MSE'] = dataframe['Valid_MSE'].map('{:.9f}'.format)
		dataframe['Valid_MAE'] = dataframe['Valid_MAE'].map('{:.9f}'.format)
		dataframe['Valid_R2' ] = dataframe['Valid_R2' ].map('{:.9f}'.format)
		dataframe['Train_MSE'] = dataframe['Train_MSE'].map('{:.9f}'.format)

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

		dataframe['Valid_Entropy' ] = dataframe['Valid_Entropy' ].map('{:.9f}'.format)
		dataframe['Valid_Accuracy'] = dataframe['Valid_Accuracy'].map('{:.9f}'.format)
		dataframe['Valid_F1'      ] = dataframe['Valid_F1'      ].map('{:.9f}'.format)
		dataframe['Valid_AUROC'   ] = dataframe['Valid_AUROC'   ].map('{:.9f}'.format)
		dataframe['Train_Entropy' ] = dataframe['Train_Entropy' ].map('{:.9f}'.format)

	return dataframe, columns

def format_cnn_tune_dataframe_params (dataframe : DataFrame) -> Tuple[DataFrame, List] :
	"""
	Doc
	"""

	columns = list()

	if 'config/optimizer/momentum' in dataframe.columns :
		columns.extend([
			'Epoch', 'Optimizer', 'Learning_Rate', 'Momentum',
			'Decay', 'Scheduler', 'Batch_Size',    'Dropout'
		])

		dataframe = dataframe.rename(columns = {
			'training_iteration'        : 'Epoch',
			'config/optimizer/name'     : 'Optimizer',
			'config/optimizer/lr'       : 'Learning_Rate',
			'config/optimizer/momentum' : 'Momentum',
			'config/optimizer/decay'    : 'Decay',
			'config/scheduler/name'     : 'Scheduler',
			'config/dataset/batch_size' : 'Batch_Size',
			'config/model/dropout'      : 'Dropout'
		})

		dataframe = dataframe.astype({
			'Epoch'         : int,
			'Optimizer'     : str,
			'Learning_Rate' : float,
			'Momentum'      : float,
			'Decay'         : float,
			'Scheduler'     : str,
			'Batch_Size'    : int,
			'Dropout'       : float
		})

		dataframe['Momentum'     ] = dataframe['Momentum'     ].map('{:.9f}'.format)
		dataframe['Learning_Rate'] = dataframe['Learning_Rate'].map('{:.9f}'.format)
		dataframe['Decay'        ] = dataframe['Decay'        ].map('{:.9f}'.format)
		dataframe['Dropout'      ] = dataframe['Dropout'      ].map('{:.3f}'.format)

	if 'config/optimizer/beta1' in dataframe.columns and 'config/optimizer/beta2' in dataframe.columns :
		columns.extend([
			'Epoch', 'Optimizer', 'Learning_Rate', 'Beta1',  'Beta2',
			'Decay', 'Scheduler', 'Batch_Size',    'Dropout'
		])

		dataframe = dataframe.rename(columns = {
			'training_iteration'        : 'Epoch',
			'config/optimizer/name'     : 'Optimizer',
			'config/optimizer/lr'       : 'Learning_Rate',
			'config/optimizer/beta1'    : 'Beta1',
			'config/optimizer/beta2'    : 'Beta2',
			'config/optimizer/decay'    : 'Decay',
			'config/scheduler/name'     : 'Scheduler',
			'config/dataset/batch_size' : 'Batch_Size',
			'config/model/dropout'      : 'Dropout'
		})

		dataframe = dataframe.astype({
			'Epoch'         : int,
			'Optimizer'     : str,
			'Learning_Rate' : float,
			'Beta1'         : float,
			'Beta2'         : float,
			'Decay'         : float,
			'Scheduler'     : str,
			'Batch_Size'    : int,
			'Dropout'       : float
		})

		dataframe['Beta1'        ] = dataframe['Beta1'        ].map('{:.9f}'.format)
		dataframe['Beta2'        ] = dataframe['Beta2'        ].map('{:.9f}'.format)
		dataframe['Learning_Rate'] = dataframe['Learning_Rate'].map('{:.9f}'.format)
		dataframe['Decay'        ] = dataframe['Decay'        ].map('{:.9f}'.format)
		dataframe['Dropout'      ] = dataframe['Dropout'      ].map('{:.3f}'.format)

	return dataframe, columns

def format_cnn_tune_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = list()

	for column in ['Model', 'Sequence', 'Trials', 'Epochs', 'Target0', 'Target1', 'Target2'] :
		if column in dataframe.columns :
			columns.append(column)

	dataframe, columns_metric = format_cnn_tune_dataframe_metrics(dataframe = dataframe, mode = mode)
	dataframe, columns_params = format_cnn_tune_dataframe_params (dataframe = dataframe)

	columns.extend(columns_metric)
	columns.extend(columns_params)

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
			'Epoch', 'Lambda'
		]

		dataframe = dataframe.rename(columns = {
			'logdir'               : 'ID',
			'valid_loss'           : 'Valid_MSE',
			'valid_mae'            : 'Valid_MAE',
			'valid_r2'             : 'Valid_R2',
			'train_loss'           : 'Train_MSE',
			'training_iteration'   : 'Epoch',
			'config/boxcox/lambda' : 'Lambda'
		})

		dataframe = dataframe.astype({
			'ID'        : str,
			'Valid_MSE' : float,
			'Valid_MAE' : float,
			'Valid_R2'  : float,
			'Train_MSE' : float,
			'Epoch'     : int,
			'Lambda'    : float,
		})

		dataframe['Valid_MSE'] = dataframe['Valid_MSE'].map('{:.9f}'.format)
		dataframe['Valid_MAE'] = dataframe['Valid_MAE'].map('{:.9f}'.format)
		dataframe['Valid_R2' ] = dataframe['Valid_R2' ].map('{:.9f}'.format)
		dataframe['Train_MSE'] = dataframe['Train_MSE'].map('{:.9f}'.format)
		dataframe['Lambda'   ] = dataframe['Lambda'   ].map('{:.9f}'.format)

	if mode == 'classification' :
		columns = [
			'ID',
			'Valid_Entropy', 'Valid_Accuracy', 'Valid_F1', 'Valid_AUROC',
			'Train_Entropy',
			'Epoch', 'Lambda', 'Bins'
		]

		dataframe = dataframe.rename(columns = {
			'logdir'               : 'ID',
			'valid_loss'           : 'Valid_Entropy',
			'valid_accuracy'       : 'Valid_Accuracy',
			'valid_f1'             : 'Valid_F1',
			'valid_auroc'          : 'Valid_AUROC',
			'train_loss'           : 'Train_Entropy',
			'training_iteration'   : 'Epoch',
			'config/boxcox/lambda' : 'Lambda',
			'config/class/bins'    : 'Bins'
		})

		dataframe = dataframe.astype({
			'ID'             : str,
			'Valid_Entropy'  : float,
			'Valid_Accuracy' : float,
			'Valid_F1'       : float,
			'Valid_AUROC'    : float,
			'Train_Entropy'  : float,
			'Epoch'          : int,
			'Lambda'         : float,
			'Bins'           : int
		})

		dataframe['Valid_Entropy' ] = dataframe['Valid_Entropy' ].map('{:.9f}'.format)
		dataframe['Valid_Accuracy'] = dataframe['Valid_Accuracy'].map('{:.9f}'.format)
		dataframe['Valid_F1'      ] = dataframe['Valid_F1'      ].map('{:.9f}'.format)
		dataframe['Valid_AUROC'   ] = dataframe['Valid_AUROC'   ].map('{:.9f}'.format)
		dataframe['Train_Entropy' ] = dataframe['Train_Entropy' ].map('{:.9f}'.format)
		dataframe['Lambda'        ] = dataframe['Lambda'        ].map('{:.9f}'.format)

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
			'Epoch'     : int,
			'Lambda'    : float,
		})

		dataframe['Lambda'] = dataframe['Lambda'].map('{:.9f}'.format)

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

		dataframe['Lambda'] = dataframe['Lambda'].map('{:.9f}'.format)

	return dataframe, columns

def format_data_tune_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = list()

	for column in ['Model', 'Sequence', 'Trials', 'Epochs', 'Target0', 'Target1', 'Target2'] :
		if column in dataframe.columns :
			columns.append(column)

	dataframe, columns_metric = format_data_tune_dataframe_metrics(dataframe = dataframe, mode = mode)
	dataframe, columns_params = format_data_tune_dataframe_params (dataframe = dataframe, mode = mode)

	columns.extend(columns_metric)
	columns.extend(columns_params)

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
			'Layer' : int,
			'Kmer'  : int,
			'Step'  : int,
			'Epoch' : int
		})

		dataframe = dataframe[[
			'Mode', 'Arch', 'Type', 'Layer', 'Kmer', 'Sequence', 'Optimizer', 'Epochs',
			'Target0', 'Target1', 'Target2',
			'Eval_R2', 'Eval_ME', 'Eval_MAPE', 'Eval_MAE', 'Learning_Rate',
			'Step', 'Epoch'
		]]

		dataframe['Eval_R2'      ] = dataframe['Eval_R2'      ].map('{:.9f}'.format)
		dataframe['Eval_ME'      ] = dataframe['Eval_ME'      ].map('{:.9f}'.format)
		dataframe['Eval_MAPE'    ] = dataframe['Eval_MAPE'    ].map('{:.9f}'.format)
		dataframe['Eval_MAE'     ] = dataframe['Eval_MAE'     ].map('{:.9f}'.format)
		dataframe['Learning_Rate'] = dataframe['Learning_Rate'].map('{:.9f}'.format)

		dataframe = dataframe.sort_values('Eval_R2', ascending = False)

	return dataframe
