from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from typing           import Any
from typing           import Dict
from typing           import List
from typing           import Tuple

from sklearn.model_selection import train_test_split

import numpy

from source.python.cnn._encoder import generate_mapping
from source.python.cnn._encoder import one_hot_encode

class GeneDataset (Dataset) :

	def __init__ (self, names : List[str], sequences : List[str], features : List[numpy.ndarray], targets : List[numpy.ndarray], expand_dims : int = None) -> None :
		"""
		Doc
		"""

		self.names     = names
		self.sequences = sequences
		self.features  = features
		self.targets   = targets

		self.mapping = generate_mapping(
			nucleotide_order = 'ACGT',
			ambiguous_value = 'fraction'
		)

		expand = lambda x : numpy.expand_dims(x, axis = expand_dims)
		encode = lambda x : one_hot_encode(
			sequence  = x,
			mapping   = self.mapping,
			default   = None,
			transpose = True
		)

		self.sequences = [encode(x) for x in self.sequences]

		if expand_dims is not None and expand_dims >= 0 :
			self.sequences = [expand(x) for x in self.sequences]

	def __getitem__ (self, index : int) -> Tuple[str, numpy.ndarray, numpy.ndarray, numpy.ndarray] :
		"""
		Doc
		"""

		return (
			self.names[index],
			self.sequences[index],
			self.features[index],
			self.targets[index]
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

	names = sorted(list(sequences.keys()))

	# AT1G21250.1         <-  default notation [Gene . Transcript]
	# AT1G21250.1-M01.0   <- mutation notation [Gene . Transcript - MutationRate . Variant

	f1 = lambda x : sequences[x]
	f2 = lambda x : numpy.array(features[x])
	f3 = lambda x : numpy.array(targets [x.split('-')[0]])

	sequences = [f1(key) for key in names]
	features  = [f2(key) for key in names]
	targets   = [f3(key) for key in names]

	return GeneDataset(
		names       = names,
		sequences   = sequences,
		features    = features,
		targets     = targets,
		expand_dims = expand_dims
	)

def generate_split_indices (targets : List[Any], test_split : float, valid_split : float, random_seed : int = None) -> Tuple[List, List | None, List | None] :
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
		targets     = dataset.targets,
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

def show_dataloader (dataloader : DataLoader, batch_size : int) -> None :
	"""
	Doc
	"""

	nbatches = len(dataloader)
	nsamples = nbatches * batch_size

	print(f'Dataloader  batch  size : {batch_size:6,d}')
	print(f'Dataloader  batch count : {nbatches:6,d}')
	print(f'Dataloader sample count : {nsamples:6,d}')
	print()

	for batch in dataloader :
		t_keys, t_sequences, t_features, t_targets = batch

		print(f'     Key shape : {numpy.shape(t_keys)}')
		print(f'Sequence shape : {numpy.shape(t_sequences)}')
		print(f' Feature shape : {numpy.shape(t_features)}')
		print(f'  Target shape : {numpy.shape(t_targets)}')

		break
