from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from typing           import Callable
from typing           import Dict
from typing           import List

import numpy

from source.python.dataset.dataset_classes import GeneDataset

def to_gene_dataset (sequences : Dict[str, str], features : Dict[str, List], targets : Dict[str, List], expand_dims : int = None, groups : Dict[str, int] = None, onehot : bool = True) -> GeneDataset :
	"""
	Doc
	"""

	names = sorted(list(features.keys()))

	transcript_key = lambda x : x.split('?')[-1].split('-')[0]

	if groups is None :
		groups = [transcript_key(x) for x in names]
	else :
		groups = [groups[transcript_key(x)] for x in names]

	return GeneDataset(
		names       = names,
		sequences   = sequences,
		features    = {k : numpy.array(v) for k, v in features.items()},
		targets     = {k : numpy.array(v) for k, v in targets.items()},
		groups      = groups,
		expand_dims = expand_dims,
		onehot      = onehot
	)

def to_dataloaders (dataset : Dataset, generator : Callable, split_size : Dict[str, float], batch_size : Dict[str, int], random_seed : int = None) -> List[DataLoader] :
	"""
	Doc
	"""

	generator = generator(
		dataset     = dataset,
		split_size  = split_size,
		random_seed = random_seed
	)

	indices = next(generator)

	train_dataloader = to_dataloader(dataset = dataset, batch_size = batch_size['train'], indices = indices[0])
	valid_dataloader = None
	test_dataloader  = None

	if indices[1] is not None : valid_dataloader = to_dataloader(dataset = dataset, batch_size = batch_size['valid'], indices = indices[1])
	if indices[2] is not None :  test_dataloader = to_dataloader(dataset = dataset, batch_size = batch_size['test'],  indices = indices[2])

	return [train_dataloader, valid_dataloader, test_dataloader]

def to_dataloader (dataset : Dataset, batch_size : int, indices : List[int]) -> DataLoader :
	"""
	Doc
	"""

	return DataLoader(
		dataset    = dataset,
		batch_size = batch_size,
		sampler    = SubsetRandomSampler(indices = indices),
		drop_last  = True
	)

def show_dataloader (dataloader : DataLoader, verbose : bool = True) -> None :
	"""
	Doc
	"""

	if not verbose :
		return

	batch_size = 0

	for batch in dataloader :
		t_keys, t_sequences, t_features, t_targets = batch

		batch_size = numpy.shape(t_keys)[0]

		print(f'     Key Shape : {numpy.shape(t_keys)}')
		print(f'Sequence Shape : {numpy.shape(t_sequences)}')
		print(f' Feature Shape : {numpy.shape(t_features)}')
		print(f'  Target Shape : {numpy.shape(t_targets)}')

		break

	nbatches = len(dataloader)
	nsamples = nbatches * batch_size

	print()
	print(f' Batch Size  : {batch_size:6,d}')
	print(f' Batch Count : {nbatches:6,d}')
	print(f'Sample Count : {nsamples:6,d}')