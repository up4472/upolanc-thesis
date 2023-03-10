from transformers.data.metrics import simple_accuracy             # noqa F821 :: unresolved reference :: added at runtime
from transformers.data.metrics import acc_and_f1                  # noqa F821 :: unresolved reference :: added at runtime
from transformers.data.metrics import acc_f1_mcc                  # noqa F821 :: unresolved reference :: added at runtime
from transformers.data.metrics import acc_f1_mcc_auc_aupr_pre_rec # noqa F821 :: unresolved reference :: added at runtime
from transformers.data.metrics import acc_f1_mcc_auc_pre_rec      # noqa F821 :: unresolved reference :: added at runtime
from transformers.data.metrics import pearson_and_spearman        # noqa F821 :: unresolved reference :: added at runtime

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error

def r2_mse_mae_me_mape (ypred, ytrue, multioutput = 'uniform_average') :
	"""
	Doc
	"""

	r2   = r2_score(y_true = ytrue, y_pred = ypred, multioutput = multioutput)
	mse  = mean_squared_error(y_true = ytrue, y_pred = ypred, multioutput = multioutput)
	mae  = mean_absolute_error(y_true = ytrue, y_pred = ypred, multioutput = multioutput)
	me   = max_error(y_true = ytrue, y_pred = ypred)
	mape = mean_absolute_percentage_error(y_true = ytrue, y_pred = ypred, multioutput = multioutput)

	return {
		'r2'        : r2,
		'max_error' : me,
		'mape'      : mape,
		'mae'       : mae,
		'mse'       : mse
	}

def compute_metrics (task_name, preds, labels, probs = None) :
	"""
	transformers.data.metrics.__init__.glue_compute_metrics()
	"""

	assert len(preds) == len(labels)

	if task_name == 'cola'       : return {'mcc' : matthews_corrcoef(labels, preds)}
	if task_name == 'sst-2'      : return {'acc' : simple_accuracy(preds, labels)}
	if task_name == 'dna690'     : return acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs)
	if task_name == 'dnapair'    : return acc_f1_mcc_auc_aupr_pre_rec(preds, labels, probs)
	if task_name == 'dnaprom'    : return acc_f1_mcc_auc_pre_rec(preds, labels, probs)
	if task_name == 'dnasplice'  : return acc_f1_mcc_auc_pre_rec(preds, labels, probs)
	if task_name == 'mrpc'       : return acc_and_f1(preds, labels)
	if task_name == 'sts-b'      : return pearson_and_spearman(preds, labels)
	if task_name == 'qqp'        : return acc_and_f1(preds, labels)
	if task_name == 'mnli'       : return {'acc' : simple_accuracy(preds, labels)}
	if task_name == 'mnli-mm'    : return {'acc' : simple_accuracy(preds, labels)}
	if task_name == 'qnli'       : return {'acc' : simple_accuracy(preds, labels)}
	if task_name == 'rte'        : return {'acc' : simple_accuracy(preds, labels)}
	if task_name == 'wnli'       : return {'acc' : simple_accuracy(preds, labels)}
	if task_name == 'hans'       : return {'acc' : simple_accuracy(preds, labels)}
	if task_name == 'regression' : return r2_mse_mae_me_mape(preds, labels)

	raise KeyError(task_name)
