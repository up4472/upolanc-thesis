from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data        import DataLoader
from torch.utils.data        import Dataset
from torch.utils.data        import SubsetRandomSampler
from typing                  import Any
from typing                  import Callable
from typing                  import Dict
from typing                  import List
from typing                  import Tuple

import numpy

from source.python.cnn._encoder import generate_mapping
from source.python.cnn._encoder import one_hot_encode

class GeneDataset (Dataset) :

	def __init__ (self, names : List[str], sequences : Dict[str, str], features : Dict[str, numpy.ndarray], targets : Dict[str, numpy.ndarray], groups : List[Any] = None, expand_dims : int = None) -> None :
		"""
		Doc
		"""

		self.names     = names
		self.sequences = sequences
		self.features  = features
		self.targets   = targets
		self.groups    = groups

		self.mapping = generate_mapping(
			nucleotide_order = 'ACGT',
			ambiguous_value  = 'fraction'
		)

		expand = lambda x : numpy.expand_dims(x, axis = expand_dims)
		encode = lambda x : one_hot_encode(
			sequence  = x,
			mapping   = self.mapping,
			default   = None,
			transpose = True
		)

		self.sequences = {
			key : encode(value)
			for key, value in self.sequences.items()
		}

		if expand_dims is not None and expand_dims >= 0 :
			self.sequences = {
				key : expand(value)
				for key, value in self.sequences.items()
			}

	def __getitem__ (self, index : int) -> Tuple[str, numpy.ndarray, numpy.ndarray, numpy.ndarray] :
		"""
		Doc
		"""

		key = self.names[index]

		#      AT1G21250.1         <-  default notation [        Gene . Transcript                         ]
		# root?AT1G21250.1         <-    group notation [Group ? Gene . Transcript                         ]
		# root?AT1G21250.1-M01.0   <- mutation notation [Group ? Gene . Transcript - MutationRate . Variant]

		# features = deafult  || sequences = deafult  || targets = deafult
		# features = group    || sequences = deafult  || targets = group
		# features = mutation || sequences = mutation || targets = group
		#                     ||  without group        || without mutation

		key_without_group    = key.split('?')[-1]
		key_without_mutation = key.split('-')[ 0]

		return (
			key,
			self.sequences[key_without_group],
			self.features [key],
			self.targets  [key_without_mutation]
		)

	def __len__ (self) -> int :
		"""
		Doc
		"""

		return len(self.features)

def to_dataset (sequences : Dict[str, str], features : Dict[str, List], targets : Dict[str, List], expand_dims : int, groups : Dict[str, int] = None) -> GeneDataset :
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
		expand_dims = expand_dims
	)

def generate_stratified_shuffle_split (dataset : GeneDataset, split_size : Dict[str, float], random_seed : int = None) -> Any :
	"""
	Doc
	"""

	tt = StratifiedShuffleSplit(test_size = split_size['test'],  n_splits = 10, random_state = random_seed)
	tv = StratifiedShuffleSplit(test_size = split_size['valid'], n_splits =  1, random_state = random_seed)

	groups = dataset.groups

	if split_size['test'] == 0.0 :
		yield numpy.arange(len(groups)), None, None

	else :
		for train_valid_index, test_index in tt.split(X = groups, y = groups) :
			if split_size['valid'] == 0.0 :
				yield train_valid_index, None, test_index

			else :
				igroups = [groups[x] for x in train_valid_index]

				train_index , valid_index = next(tv.split(X = igroups, y = igroups))

				train_index = train_valid_index[train_index]
				valid_index = train_valid_index[valid_index]

				yield train_index, valid_index, test_index

def generate_group_shuffle_split (dataset : GeneDataset, split_size : Dict[str, float], random_seed : int = None) -> Any :
	"""
	Doc
	"""

	tt = GroupShuffleSplit(test_size = split_size['test'],  n_splits = 10, random_state = random_seed)
	tv = GroupShuffleSplit(test_size = split_size['valid'], n_splits =  1, random_state = random_seed)

	groups = dataset.groups

	if split_size['test'] == 0.0 :
		yield numpy.arange(len(groups)), None, None

	else :
		for train_valid_index, test_index in tt.split(X = groups, groups = groups) :
			if split_size['valid'] == 0.0 :
				yield train_valid_index, None, test_index

			else :
				igroups = [groups[x] for x in train_valid_index]

				train_index , valid_index = next(tv.split(X = igroups, groups = igroups))

				train_index = train_valid_index[train_index]
				valid_index = train_valid_index[valid_index]

				yield train_index, valid_index, test_index

def generate_shuffle_split (dataset : GeneDataset, split_size : Dict[str, float], random_seed : int = None) -> Any :
	"""
	Doc
	"""

	tt = ShuffleSplit(test_size = split_size['test'],  n_splits = 10, random_state = random_seed)
	tv = ShuffleSplit(test_size = split_size['valid'], n_splits =  1, random_state = random_seed)

	groups = dataset.groups

	if split_size['test'] == 0.0 :
		yield numpy.arange(len(groups)), None, None

	else :
		for train_valid_index, test_index in tt.split(X = groups) :
			if split_size['valid'] == 0.0 :
				yield train_valid_index, None, test_index

			else :
				igroups = [groups[x] for x in train_valid_index]

				train_index , valid_index = next(tv.split(X = igroups))

				train_index = train_valid_index[train_index]
				valid_index = train_valid_index[valid_index]

				yield train_index, valid_index, test_index

def to_dataloaders (dataset : GeneDataset, generator : Callable, split_size : Dict[str, float], batch_size : Dict[str, int], random_seed : int = None) -> List[DataLoader] :
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

def to_dataloader (dataset : GeneDataset, batch_size : int, indices : List[int]) -> DataLoader :
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
