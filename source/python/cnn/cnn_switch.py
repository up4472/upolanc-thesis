from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from torch.utils.data import DataLoader

from source.python.dataset.dataset_classes import GeneDataset
from source.python.dataset.dataset_utils import to_dataloaders
from source.python.dataset.dataset_utils   import to_gene_dataset

def append_drop_to_each (keep : List[List], drop : List[Dict], full : List[Dict]) -> Tuple[List, List] :
	"""
	Doc
	"""

	x = list()
	y = list()
	z = list()

	for index in range(len(keep[0])) :
		xi = {k : v for k, v in keep[0][index].items()}
		yi = {k : v for k, v in keep[1][index].items()}
		zi = {k : v for k, v in keep[2][index].items()}

		xi.update(drop[0])
		yi.update(drop[1])
		zi.update(drop[2])

		x.append(xi)
		y.append(yi)
		z.append(zi)

		a = len(keep[2][index])
		b = len(drop[2])

		print('Majority Vote [#{:02d}] : {:.5f}'.format(index, max(a, b) / (a + b)))

	full[0].update(drop[0])
	full[1].update(drop[1])
	full[2].update(drop[2])

	a = len(full[2])
	b = len(drop[2])

	print('Majority Vote [All] : {:.5f}'.format(max(a, b) / (a + b)))
	print()

	return [x, y, z], full

def to_dataset (sequence : Union[List, Dict], features : Union[List, Dict], targets : Union[List, Dict], config : Dict[str, Any]) -> List[GeneDataset] :
	"""
	Doc
	"""

	if not isinstance(sequence, list) : sequence = [sequence]
	if not isinstance(features, list) : features = [features]
	if not isinstance(targets,  list) : targets  = [targets]

	datasets = list()

	expand = config['dataset/expanddim']
	start  = config['dataset/sequence/start']
	end    = config['dataset/sequence/end']

	for index in range(len(sequence)) :
		dataset = to_gene_dataset(
			sequences   = sequence[index],
			features    = features[index],
			targets     = targets[index],
			expand_dims = expand,
			groups      = None,
			start       = start,
			end         = end
		)

		if start is None : start = 0
		if end   is None : end   = len(list(sequence[index].values())[0])
		if start >= end  : raise ValueError()

		datasets.append(dataset)

	config['dataset/sequence/start'] = start
	config['dataset/sequence/end']   = end
	config['model/input/width']      = int(end - start)

	return datasets

def to_dataloader (dataset_all : GeneDataset, dataset_mix : List[GeneDataset], config : Dict[str, Any]) -> Tuple[DataLoader, List[List]] :
	"""
	Doc
	"""

	dataloader_all = to_dataloaders(
		dataset     = dataset_all,
		generator   = config['dataset/split/generator'],
		random_seed = config['core/random'],
		split_size  = {
			'valid' : 0.0,
			'test'  : 0.0
		},
		batch_size  = {
			'train' : config['dataset/batch/train'],
			'valid' : config['dataset/batch/valid'],
			'test'  : config['dataset/batch/test']
		}
	)

	train_dataloader = list()
	valid_dataloader = list()
	test_dataloader  = list()

	for index in range(len(dataset_mix)) :
		dataloader = to_dataloaders(
			dataset     = dataset_mix[index],
			generator   = config['dataset/split/generator'],
			random_seed = config['core/random'],
			split_size  = {
				'valid' : config['dataset/split/valid'],
				'test'  : config['dataset/split/test']
			},
			batch_size  = {
				'train' : config['dataset/batch/train'],
				'valid' : config['dataset/batch/valid'],
				'test'  : config['dataset/batch/test']
			}
		)

		train_dataloader.append(dataloader[0])
		valid_dataloader.append(dataloader[1])
		test_dataloader.append(dataloader[2])

	return dataloader_all[0], [train_dataloader, valid_dataloader, test_dataloader]
