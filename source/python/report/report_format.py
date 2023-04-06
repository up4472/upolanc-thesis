from pandas import DataFrame

def format_tune_model_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = []
	custom  = [
		'Model', 'Sequence', 'Trials', 'Epochs',
		'Target0', 'Target1', 'Target2'
	]

	for column in custom :
		if column in dataframe.columns :
			columns.append(column)

	if mode == 'regression' :
		columns.extend([
			'logdir',
			'valid_loss', 'valid_mae', 'valid_r2',
			'train_loss',
			'training_iteration',
			'config/optimizer/name', 'config/optimizer/lr', 'config/optimizer/momentum', 'config/optimizer/decay',
			'config/scheduler/name', 'config/dataset/batch_size', 'config/model/dropout'
		])

		dataframe = dataframe[columns].copy()

		dataframe = dataframe.rename(columns = {
			'logdir'                    : 'ID',
			'valid_loss'                : 'Valid_MSE',
			'valid_mae'                 : 'Valid_MAE',
			'valid_r2'                  : 'Valid_R2',
			'train_loss'                : 'Train_MSE',
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
			'ID'            : str,
			'Valid_MSE'     : float,
			'Valid_MAE'     : float,
			'Valid_R2'      : float,
			'Train_MSE'     : float,
			'Epoch'         : int,
			'Optimizer'     : str,
			'Learning_Rate' : float,
			'Momentum'      : float,
			'Decay'         : float,
			'Scheduler'     : str,
			'Batch_Size'    : int,
			'Dropout'       : float
		})

		dataframe['Valid_MSE'    ] = dataframe['Valid_MSE'    ].map('{:.9f}'.format)
		dataframe['Valid_MAE'    ] = dataframe['Valid_MAE'    ].map('{:.9f}'.format)
		dataframe['Valid_R2'     ] = dataframe['Valid_R2'     ].map('{:.9f}'.format)
		dataframe['Train_MSE'    ] = dataframe['Train_MSE'    ].map('{:.9f}'.format)
		dataframe['Learning_Rate'] = dataframe['Learning_Rate'].map('{:.9f}'.format)
		dataframe['Momentum'     ] = dataframe['Momentum'     ].map('{:.9f}'.format)
		dataframe['Decay'        ] = dataframe['Decay'        ].map('{:.9f}'.format)
		dataframe['Dropout'      ] = dataframe['Dropout'      ].map('{:.3f}'.format)

	if mode == 'classification' :
		columns.extend([
			'logdir',
			'valid_loss', 'valid_accuracy', 'valid_f1', 'valid_auroc',
			'train_loss',
			'training_iteration',
			'config/optimizer/name', 'config/optimizer/lr', 'config/optimizer/momentum', 'config/optimizer/decay',
			'config/scheduler/name', 'config/dataset/batch_size', 'config/model/dropout'
		])

		dataframe = dataframe[columns].copy()

		dataframe = dataframe.rename(columns = {
			'logdir'                    : 'ID',
			'valid_loss'                : 'Valid_Entropy',
			'valid_accuracy'            : 'Valid_Accuracy',
			'valid_f1'                  : 'Valid_F1',
			'valid_auroc'               : 'Valid_AUROC',
			'train_loss'                : 'Train_Entropy',
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
			'ID'             : str,
			'Valid_Entropy'  : float,
			'Valid_Accuracy' : float,
			'Valid_F1'       : float,
			'Valid_AUROC'    : float,
			'Train_Entropy'  : float,
			'Epoch'          : int,
			'Optimizer'      : str,
			'Learning_Rate'  : float,
			'Momentum'       : float,
			'Decay'          : float,
			'Scheduler'      : str,
			'Batch_Size'     : int,
			'Dropout'        : float
		})

		dataframe['Valid_Entropy' ] = dataframe['Valid_Entropy' ].map('{:.9f}'.format)
		dataframe['Valid_Accuracy'] = dataframe['Valid_Accuracy'].map('{:.9f}'.format)
		dataframe['Valid_F1'      ] = dataframe['Valid_F1'      ].map('{:.9f}'.format)
		dataframe['Valid_AUROC'   ] = dataframe['Valid_AUROC'   ].map('{:.9f}'.format)
		dataframe['Train_Entropy' ] = dataframe['Train_Entropy' ].map('{:.9f}'.format)
		dataframe['Learning_Rate' ] = dataframe['Learning_Rate' ].map('{:.9f}'.format)
		dataframe['Momentum'      ] = dataframe['Momentum'      ].map('{:.9f}'.format)
		dataframe['Decay'         ] = dataframe['Decay'         ].map('{:.9f}'.format)
		dataframe['Dropout'       ] = dataframe['Dropout'       ].map('{:.3f}'.format)

	return dataframe

def format_tune_data_dataframe (dataframe : DataFrame, mode : str) -> DataFrame :
	"""
	Doc
	"""

	columns = []
	custom  = [
		'Model', 'Sequence', 'Trials', 'Epochs',
		'Target0', 'Target1', 'Target2'
	]

	for column in custom :
		if column in dataframe.columns :
			columns.append(column)

	if mode == 'regression' :
		columns.extend([
			'logdir',
			'valid_loss', 'valid_mae', 'valid_r2',
			'train_loss',
			'training_iteration',
			'config/boxcox/lambda'
		])

		dataframe = dataframe[columns].copy()

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
		columns.extend([
			'logdir',
			'valid_loss', 'valid_accuracy', 'valid_f1', 'valid_auroc',
			'train_loss',
			'training_iteration',
			'config/boxcox/lambda', 'config/class/bins'

		])

		dataframe = dataframe[columns].copy()

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
			'Step' : int
		})

		dataframe = dataframe[[
			'Mode', 'Model', 'Kmer', 'Sequence', 'Epochs', 'Target0', 'Target1', 'Target2',
			'Eval_R2', 'Eval_ME', 'Eval_MAPE', 'Eval_MAE', 'Learning_Rate', 'Step'
		]]

		dataframe['Eval_R2'      ] = dataframe['Eval_R2'      ].map('{:.9f}'.format)
		dataframe['Eval_ME'      ] = dataframe['Eval_ME'      ].map('{:.9f}'.format)
		dataframe['Eval_MAPE'    ] = dataframe['Eval_MAPE'    ].map('{:.9f}'.format)
		dataframe['Eval_MAE'     ] = dataframe['Eval_MAE'     ].map('{:.9f}'.format)
		dataframe['Learning_Rate'] = dataframe['Learning_Rate'].map('{:.9f}'.format)

		dataframe = dataframe.sort_values('Eval_R2', ascending = False)

	return dataframe
