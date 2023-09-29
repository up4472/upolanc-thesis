from torch.utils.data import DataLoader
from typing           import Any
from typing           import Dict
from typing           import List

import matplotlib
import numpy

from source.python.dataset.dataset_utils import to_dataloader
from source.python.dataset.dataset_utils import to_gene_dataset

def create_dataloader (sequences : Dict[str, str], features : Dict[str, List], targets : Dict[str, List], expand_dims : int = None, start : int = None, end : int = None) -> DataLoader :
	"""
	Doc
	"""

	dataset = to_gene_dataset(
		sequences   = sequences,
		features    = features,
		targets     = targets,
		expand_dims = expand_dims,
		start       = start,
		end         = end
	)

	return to_dataloader(
		dataset     = dataset,
		batch_size  = 1,
		indices     = [i for i in range(len(dataset))]
	)

def get_mutation_report (report : Dict[str, Dict]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	keys  = report['eval']['keys']
	ypred = report['eval']['ypred']
	ytrue = report['eval']['ytrue']

	data = dict()

	for index, key in enumerate(keys) :
		b0 = key.split('-')[0]
		b1 = key.split('-')[1].split('.')[0]

		if b0 not in data.keys() :
			data[b0] = dict()

		if b1 not in data[b0].keys() :
			data[b0][b1] = {
				'ypred' : list(),
				'ytrue' : ytrue[index, :],
				'label' : list()
			}

		data[b0][b1]['ypred'].append(ypred[index, :])
		data[b0][b1]['label'].append(key)

	return data

def plot_mutation_classification (report : Dict[str, Dict], order : List[str], transcript : str = None, mutation : str = None, alpha : float = 0.9, linewidth : int = 2, filename : str = None) -> None : # noqa : unused
	"""
	Doc
	"""

	if transcript is None :
		transcript = list(report.keys())[0]
	if mutation is None :
		mutation = list(report[transcript].keys())[0]

	data = report[transcript][mutation]
	ndim = numpy.ndim(data['ypred'][0])

	if ndim <= 1 :
		return

	print('TODO - to be implemented')
	return

def plot_mutation_regression (report : Dict[Any, Any], order : Any, transcript : Any, mutation : Any, filename : str = None) -> None :
	"""
	Doc
	"""

	if transcript is None :
		transcript = list(report.keys())[0]
	if mutation is None :
		mutation = list(report[transcript].keys())[0]

	data = report[transcript][mutation]
	ndim = numpy.ndim(data['ypred'][0])
	nmut = len(data['ypred'])

	if ndim > 1 :
		return

	fig, ax = matplotlib.pyplot.subplots(figsize = (16, 10))
	fig.tight_layout()

	ytrue = report[transcript]['M00']['ytrue']
	ypred = report[transcript]['M00']['ypred'][0]

	if ypred.size == 1 :
		y = [numpy.array(ypred).flatten() - ytrue] * nmut
		x = numpy.arange(nmut)

		ax.plot(x, y, linestyle = '-', linewidth = 4, color = 'g', alpha = 0.9)

		y = numpy.array(data['ypred']).flatten() - ytrue
		ax.plot(x, y, linestyle = '-', linewidth = 2, color = 'b', alpha = 0.4)

		ax.set_xticks(range(nmut))
		ax.set_xlabel('Variant')
		ax.set_ylabel('MAE(TPM)')
	else :
		ax.plot(ytrue, linestyle = '-', linewidth = 4, color = 'r', alpha = 0.9)
		ax.plot(ypred, linestyle = '-', linewidth = 4, color = 'g', alpha = 0.9)

		for y, label in zip(data['ypred'], data['label']) :
			ax.plot(y, linestyle = '-', linewidth = 2, color = 'b', alpha = 0.4)

		ax.set_xticks(range(len(order)))
		ax.set_xticklabels(order)
		ax.set_xlabel('Group')
		ax.set_ylabel('TPM')

	ax.set_title('{} with MR = {:.2f}'.format(transcript, int(mutation[1:]) / 100))
	ax.legend(['Ground Truth', 'Predicted (Pure)', 'Predicted (Variant)'])

	if filename is not None :
		matplotlib.pyplot.savefig(
			'{}-{}-{}.png'.format(filename, transcript, mutation).replace('?', '_'),
			dpi         = 120,
			format      = 'png',
			bbox_inches = 'tight',
			pad_inches  = 0
		)
