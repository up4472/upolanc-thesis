from torch.utils.data import Dataset
from typing           import Any
from typing           import Dict
from typing           import List
from typing           import Tuple

import numpy

from source.python.encoding.onehot import generate_onehot_mapping
from source.python.encoding.onehot import onehot_encode

class GeneDataset (Dataset) :

	def __init__ (self, names : List[str], sequences : Dict[str, str], features : Dict[str, numpy.ndarray], targets : Dict[str, numpy.ndarray], groups : List[Any] = None, expand_dims : int = None, onehot : bool = True) -> None :
		"""
		Doc
		"""

		self.names     = names
		self.sequences = sequences
		self.features  = features
		self.targets   = targets
		self.groups    = groups

		self.mapping = generate_onehot_mapping(
			nucleotide_order = 'ACGT',
			ambiguous_value  = 'fraction'
		)

		expand = lambda x : numpy.expand_dims(x, axis = expand_dims)
		encode = lambda x : onehot_encode(
			sequence  = x,
			mapping   = self.mapping,
			default   = None,
			transpose = True
		)

		if onehot :
			self.sequences = {
				key : encode(value)
				for key, value in self.sequences.items()
			}

			if expand_dims is not None and expand_dims >= 0 :
				self.sequences = {
					key : expand(value)
					for key, value in self.sequences.items()
				}

	def __getitem__ (self, index : int) -> Tuple[str, Any, numpy.ndarray, numpy.ndarray] :
		"""
		Doc
		"""

		key = self.names[index]

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
