from torch.utils.data import DataLoader
from typing           import Dict
from typing           import List

import matplotlib
import numpy

from source.python.cnn.dataset import to_dataloaders
from source.python.cnn.dataset import to_dataset

def create_dataloader (sequences : Dict[str, str], features : Dict[str, List], targets : Dict[str, List], expand_dims : int = None, random_seed : int = None) -> DataLoader :
	"""
	Doc
	"""

	dataset = to_dataset(
		sequences   = sequences,
		features    = features,
		targets     = targets,
		expand_dims = expand_dims
	)

	return to_dataloaders(
		dataset     = dataset,
		random_seed = random_seed,
		split_size  = { 'valid' : 0, 'test' : 0 },
		batch_size  = { 'train' : 1 }
	)[0]

def get_mutation_report (report : Dict[str, Dict]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	label = report['eval']['genes']
	ypred = report['eval']['ypred']
	ytrue = report['eval']['ytrue']

	data = dict()

	for index, transcript in enumerate(label) :
		b0 = transcript.split('-')[0]
		b1 = transcript.split('-')[1].split('.')[0]

		if b0 not in data.keys() :
			data[b0] = dict()

		if b1 not in data[b0].keys() :
			data[b0][b1] = {
				'ypred' : list(),
				'ytrue' : ytrue[index, :],
				'label' : list()
			}

		data[b0][b1]['ypred'].append(ypred[index, :])
		data[b0][b1]['label'].append(transcript)

	return data

def plot_mutation_classification (report : Dict[str, Dict], order : List[str], transcript : str = None, mutation : str = None, filename : str = None) -> None : # noqa : unused
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

def plot_mutation_regression (report : Dict[str, Dict], order : List[str], transcript : str = None, mutation : str = None, filename : str = None) -> None :
	"""
	Doc
	"""

	if transcript is None :
		transcript = list(report.keys())[0]
	if mutation is None :
		mutation = list(report[transcript].keys())[0]

	data = report[transcript][mutation]
	ndim = numpy.ndim(data['ypred'][0])

	if ndim > 1 :
		return

	_, ax = matplotlib.pyplot.subplots(figsize = (16, 10))

	ytrue = report[transcript]['M00']['ytrue']
	ypred = report[transcript]['M00']['ypred'][0]

	ax.plot(ytrue, linestyle = '-', linewidth = 3, color = 'r', alpha = 0.9)
	ax.plot(ypred, linestyle = '-', linewidth = 3, color = 'g', alpha = 0.9)

	for variant, label in zip(data['ypred'], data['label']) :
		ax.plot(variant, linestyle = '--', linewidth = 1, color = 'b', alpha = 0.3)

	ax.set_title('{} with MR = {:.2f}'.format(transcript, int(mutation[1:]) / 100))
	ax.set_xticks(range(len(order)))
	ax.set_xticklabels(order)
	ax.legend(['Ground Truth', 'Predicted (Pure)', 'Predicted (Variant)'])

	if filename is not None :
		matplotlib.pyplot.savefig(
			'{}-{}-{}.png'.format(filename, transcript, mutation),
			dpi    = 120,
			format = 'png'
		)
