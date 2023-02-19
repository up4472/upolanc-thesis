from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from typing           import Any
from typing           import Dict
from typing           import List
from typing           import Optional
from typing           import Tuple

from sklearn.model_selection import train_test_split

import numpy

from source.python.cnn._encoder import generate_mapping
from source.python.cnn._encoder import one_hot_encode

class GeneDataset (Dataset) :

	def __init__ (self, names : List[str], sequences : Dict[str, str], features : Dict[str, numpy.ndarray], targets : Dict[str, numpy.ndarray], expand_dims : int = None) -> None :
		"""
		Doc
		"""

		self.names     = names
		self.sequences = sequences
		self.features  = features
		self.targets   = targets

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

		return len(self.targets)

def to_dataset (sequences : Dict[str, str], features : Dict[str, List], targets : Dict[str, List], expand_dims : int) -> GeneDataset :
	"""
	Doc
	"""

	names = sorted(list(features.keys()))

	return GeneDataset(
		names       = names,
		sequences   = sequences,
		features    = {k : numpy.array(v) for k, v in features.items()},
		targets     = {k : numpy.array(v) for k, v in targets.items()},
		expand_dims = expand_dims
	)

def generate_split_indices (targets : List[Any], test_split : float, valid_split : float, random_seed : int = None) -> Tuple[List, Optional[List], Optional[List]] :
	"""
	Doc
	"""

	length = len(targets)
	arange = numpy.arange(length)

	if test_split > 0 :
		s1, s3 = train_test_split(arange, random_state = random_seed, shuffle = True, stratify = None, test_size = test_split)
	else :
		s1 = arange
		s3 = None

	if valid_split > 0 :
		s1, s2 = train_test_split(s1, random_state = random_seed, shuffle = True, stratify = None, test_size = valid_split)
	else :
		s1 = s1
		s2 = None

	return s1, s2, s3

def to_dataloaders (dataset : GeneDataset, split_size : Dict[str, float], batch_size : Dict[str, int], random_seed : int = None) -> List[DataLoader] :
	"""
	Doc
	"""

	train_idx, valid_idx, test_idx = generate_split_indices(
		targets     = dataset.names,
		valid_split = split_size['valid'],
		test_split  = split_size['test'],
		random_seed = random_seed
	)

	dataloaders = [
		to_dataloader(dataset = dataset, batch_size = batch_size['train'], indices = train_idx)
	]

	if valid_idx is not None :
		dataloaders.append(
			to_dataloader(dataset = dataset, batch_size = batch_size['valid'], indices = valid_idx)
		)

	if test_idx is not None :
		dataloaders.append(
			to_dataloader(dataset = dataset, batch_size = batch_size['test'], indices = test_idx)
		)

	return dataloaders

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
