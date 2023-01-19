from torch            import Generator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split
from typing           import Dict
from typing           import List
from typing           import Tuple
import itertools
import numpy

from src.cnn._encoder import generate_mapping
from src.cnn._encoder import one_hot_encode

class GeneDataset (Dataset) :

	def __init__ (self, sequences : Dict[str, str], features : Dict[str, numpy.ndarray], targets : Dict[str, numpy.ndarray], expand_dims : int = None) -> None :
		"""
		Doc
		"""

		self.sequence = dict()
		self.features = features
		self.targets  = targets

		self.keys = list(sequences.keys())

		self.mapping = generate_mapping(
			nucleotide_order = 'ACGT',
			ambiguous_value = 'zero'
		)

		for key, value in sequences.items() :
			self.sequence[key] = one_hot_encode(
				sequence  = value,
				mapping   = self.mapping,
				default   = None,
				transpose = True
			)

		if expand_dims is not None and expand_dims >= 0 :
			for key in self.sequence.keys() :
				self.sequence[key] = numpy.expand_dims(self.sequence[key], axis = expand_dims)

	def __getitem__ (self, index : int) -> Tuple[str, numpy.ndarray, numpy.ndarray, numpy.ndarray] :
		"""
		Doc
		"""

		key = self.keys[index]

		return (
			key,
			self.sequence[key],
			self.features[key],
			self.targets[key]
		)

	def __len__ (self) -> int :
		"""
		Doc
		"""

		return len(self.keys)

	def split_to_subsets (self, split_size : List[float], random_seed : int = None) -> List[Subset] :
		"""
		Doc
		"""

		if random_seed is not None :
			generator = Generator().manual_seed(random_seed)
		else :
			generator = Generator()

		return random_split(
			dataset   = self,
			lengths   = split_size,
			generator = generator
		)

	def split_to_dataloader (self, split_size : List[float], batch_size : List[int], random_seed : int = None, **kwargs) -> List[DataLoader] :
		"""
		Doc
		"""

		subsets = self.split_to_subsets(split_size = split_size, random_seed = random_seed)

		if len(batch_size) > len(subsets) :
			batch_size = batch_size[:len(subsets)]

		return [
			DataLoader(dataset = subset, batch_size = size, **kwargs)
			for subset, size in itertools.zip_longest(subsets, batch_size, fillvalue = batch_size[-1])
		]

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
