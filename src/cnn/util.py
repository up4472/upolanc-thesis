from pandas import DataFrame
from typing import Any
from typing import Dict
from typing import List

import numpy

def display_regression_predictions (report : Dict[str, Any], n : int = 5) -> None :
	"""
	Doc
	"""

	eval_mae = report['eval']['metric']['mae']
	eval_mse = report['eval']['metric']['mse']
	eval_genes = report['eval']['genes']
	eval_ytrue = report['eval']['ytrue']
	eval_ypred = report['eval']['ypred']

	if len(eval_ytrue) > 0 and len(eval_ypred) > 0 :
		items = zip(eval_genes, eval_mae, eval_mse, eval_ytrue, eval_ypred)
	else :
		items = zip(eval_genes, eval_mae, eval_mse)

	for index, item in enumerate(items) :
		if index >= n :
			break

		gene = item[0]
		mae  = item[1]
		mse  = item[2]

		print(f'Gene : {gene}')

		if len(item) == 5 :
			ytrue = item[3]
			ypred = item[4]

			print(f'True : [' + '   '.join('{:.5f}'.format(x) for x in ytrue) + ']')
			print(f'Pred : [' + '   '.join('{:.5f}'.format(x) for x in ypred) + ']')

		print(f' MAE : [' + '   '.join('{:.5f}'.format(x) for x in mae) + ']')
		print(f' MSE : [' + '   '.join('{:.5f}'.format(x) for x in mse) + ']')
		print()

def display_classification_predictions (report : Dict[str, Any], n : int = 5) -> None :
	"""
	Doc
	"""

	eval_cce = report['eval']['metric']['entropy']
	eval_acc = report['eval']['metric']['acc']
	eval_genes = report['eval']['genes']
	eval_ytrue = report['eval']['ytrue']
	eval_ypred = report['eval']['ypred']

	if len(eval_ytrue) > 0 and len(eval_ypred) > 0 :
		items = zip(eval_genes, eval_cce, eval_acc, eval_ytrue, eval_ypred)
	else :
		items = zip(eval_genes, eval_cce, eval_acc)

	for index, item in enumerate(items) :
		if index >= n :
			break

		gene = item[0]
		cce  = item[1]
		acc  = item[2]

		print(f'Gene : {gene}')

		if len(item) == 5 :
			ytrue = item[3]
			ypred = item[4].argmax(axis = 0)

			print(f'True : [' + ' '.join('{:}'.format(x) for x in ytrue) + ']')
			print(f'Pred : [' + ' '.join('{:}'.format(x) for x in ypred) + ']')

		print(f' CCE : [' + '   '.join('{:.5f}'.format(x) for x in cce) + ']')
		print(f' ACC : [' + '   '.join('{:.5f}'.format(x) for x in acc) + ']')
		print()

def display_regression_accuracy (report : Dict[str, Any], order : List[str], threshold : Dict[str, numpy.ndarray]) -> DataFrame :
	"""
	Doc
	"""

	genes  = report['genes']
	scores = report['metric']['mae']

	taccuracy = [score <= threshold[gene] for gene, score in zip(genes, scores)]
	taccuracy = numpy.array(taccuracy, dtype = bool)

	count = numpy.sum(taccuracy, axis = 0)
	total = numpy.shape(taccuracy)[0]

	accuracy = 100.0 * count / total

	return DataFrame.from_dict({
		'Group'    : order,
		'Total'    : total,
		'Count'    : count,
		'Accuracy' : accuracy,
		'Avg MAE'  : numpy.mean(scores, axis = 0),
		'Std MAE'  : numpy.std(scores, axis = 0)
	})

def display_classification_accuracy (report : Dict[str, Any], order : List[str]) -> DataFrame :
	"""
	Doc
	"""

	accuracy = report['metric']['acc']

	ypred = report['ypred']
	ytrue = report['ytrue']

	total = numpy.shape(accuracy)[0]
	accuracy = accuracy.mean(axis = 0) * 100.0

	ypred = [x.argmax(axis = 0) for x in ypred]
	ydiff = numpy.abs(ytrue - ypred)

	m1 = numpy.mean(ydiff == 1, axis = 0) * 100.0
	m2 = numpy.mean(ydiff == 2, axis = 0) * 100.0
	m3 = numpy.mean(ydiff == 3, axis = 0) * 100.0
	m4 = numpy.mean(ydiff == 4, axis = 0) * 100.0

	return DataFrame.from_dict({
		'Group'    : order,
		'Total'    : total,
		'Accuracy' : accuracy,
		'Missed_1' : m1,
		'Missed_2' : m2,
		'Missed_3' : m3,
		'Missed_4' : m4
	})
