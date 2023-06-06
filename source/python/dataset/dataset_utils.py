from pandas           import DataFrame
from torch            import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from typing           import Any
from typing           import Callable
from typing           import Dict
from typing           import List
from typing           import Tuple
from typing           import Union

import numpy

from source.python.dataset.dataset_classes import GeneDataset
from source.python.dataset.dataset_split   import generate_group_shuffle_split
from source.python.dataset.dataset_split   import generate_random_shuffle_split
from source.python.dataset.dataset_split   import generate_stratified_shuffle_split
from source.python.io.loader               import load_feature_targets

def to_gene_dataset (sequences : Dict[str, Any], features : Dict[str, List], targets : Dict[str, List], expand_dims : int = None, groups : Dict[str, Union[int, float]] = None, onehot : bool = True, start : int = None, end : int = None) -> GeneDataset :
	"""
	Doc
	"""

	names = sorted(list(features.keys()))

	transcript_key = lambda x : x.split('?')[-1].split('-')[0]

	if groups is None :
		groups = [transcript_key(x) for x in names]
	else :
		groups = [int(groups[transcript_key(x)]) for x in names]

	return GeneDataset(
		names       = names,
		sequences   = sequences,
		features    = {k : numpy.array(v) for k, v in features.items()},
		targets     = {k : numpy.array(v) for k, v in targets.items()},
		groups      = groups,
		expand_dims = expand_dims,
		onehot      = onehot,
		start       = start,
		end         = end
	)

def to_dataloaders (dataset : GeneDataset, generator : Union[str, Callable], split_size : Dict[str, float], batch_size : Dict[str, int], random_seed : int = None) -> List[DataLoader] :
	"""
	Doc
	"""

	if isinstance(generator, str) :
		if   generator.startswith('stratified') : generator = generate_stratified_shuffle_split
		elif generator.startswith('group')      : generator = generate_group_shuffle_split
		elif generator.startswith('random')     : generator = generate_random_shuffle_split
		else : raise ValueError()

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
		batch_size = numpy.shape(batch[0])[0]
		batch_item = [
			('Key',      batch[0]),
			('Sequence', batch[1]),
			('Feature',  batch[2]),
			('Target',   batch[3])
		]

		for name, item in batch_item :
			dtype = '-'

			if isinstance(item, Tensor) :
				dtype = str(item.dtype)

			print('{:8s} : {:13s} | {}'.format(name, dtype, ', '.join(str(x) for x in numpy.shape(item))))

		break

	nbatches = len(dataloader)
	nsamples = nbatches * batch_size

	print()
	print(f'Batch Size : {batch_size:6,d}')
	print(f'Batches    : {nbatches:6,d}')
	print(f'Samples    : {nsamples:6,d}')
	print()

def get_dataset (config : Dict[str, Any], sequence : Dict[str, Any], feature : Dict[str, Any], directory : str, filename : str, cached : Dict[str, Any] = None, start : int = None, end : int = None) -> Tuple[GeneDataset, DataFrame, Dict, List] :
	"""
	Doc
	"""

	target_group   = config['model/output/target']
	target_type    = config['model/output/type']
	target_filter  = config['model/output/filter']
	target_explode = config['model/output/explode']

	filters = {
		'tissue'       : None,
		'group'        : None,
		'age'          : None,
		'perturbation' : None,
		'global'       : None
	} | {
		target_group : target_filter
		if target_filter is None
		else [target_filter]
	}

	dataframe, target_value, target_order = load_feature_targets(
		group     = '{}-{}'.format(target_group, target_type),
		explode   = target_explode,
		filters   = filters,
		directory = directory,
		filename  = filename,
		mode      = config['model/mode'],
		cached    = cached
	)

	if 'Feature' in dataframe.columns :
		feature = {
			key : numpy.concatenate((feature[key.split('?')[-1]], value))
			for key, value in dataframe['Feature'].to_dict().items()
		}

	if config['model/mode'] == 'regression' :
		config['model/output/size']    = len(target_order)
		config['model/input/features'] = len(list(feature.values())[0])

	if config['model/mode'] == 'classification' :
		config['model/output/size']    = len(numpy.unique(numpy.array([x for x in dataframe['TPM_Label']]).flatten()))
		config['model/output/heads']   = len(target_order)
		config['model/input/features'] = len(list(feature.values())[0])

	dataset = to_gene_dataset(
		sequences   = sequence,
		features    = feature,
		targets     = target_value,
		expand_dims = config['dataset/expanddim'],
		groups      = None,
		start       = start,
		end         = end
	)

	if start is None : start = 0
	if end   is None : end   = len(list(sequence.values())[0])

	if start >= end :
		raise ValueError()

	config['model/input/width'] = int(end - start)

	return dataset, dataframe, target_value, target_order
