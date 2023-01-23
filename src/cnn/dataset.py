from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from typing           import Any
from typing           import List
from typing           import Tuple

from sklearn.model_selection import train_test_split

import numpy

from src.cnn._encoder import generate_mapping
from src.cnn._encoder import one_hot_encode

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

def generate_split_indices (targets : List[Any], test_split : float, valid_split : float, random_seed : int = None) -> Tuple[List, List, List] :
	"""
	Doc
	"""

	length = len(targets)
	arange = numpy.arange(length)

	s1, s3 = train_test_split(arange, random_state = random_seed, shuffle = True, stratify = None, test_size = test_split)
	s1, s2 = train_test_split(s1,     random_state = random_seed, shuffle = True, stratify = None, test_size = valid_split)

	train_split = 100 * len(s1) / length
	valid_split = 100 * len(s2) / length
	test_split  = 100 * len(s3) / length

	print(f'Train percentage : {train_split:5.2f}')
	print(f'Valid percentage : {valid_split:5.2f}')
	print(f' Test percentage : {test_split:5.2f}')

	return s1, s2, s3

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
