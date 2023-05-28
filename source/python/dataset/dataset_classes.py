from torch.utils.data import Dataset
from typing           import Any
from typing           import Dict
from typing           import List
from typing           import Tuple

import numpy

from source.python.dataset.dataset_sequence import get_subsequences
from source.python.dataset.dataset_sequence import get_encoding
from source.python.encoding.onehot          import generate_onehot_mapping
from source.python.encoding.onehot          import onehot_encode

class GeneDataset (Dataset) :

	def __init__ (self, names : List[str], sequences : Dict[str, Any], features : Dict[str, numpy.ndarray], targets : Dict[str, numpy.ndarray], groups : List[Any] = None, expand_dims : int = None, onehot : bool = True, start : int = None, end : int = None) -> None :
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

		self.sequences = get_subsequences(
			sequences = self.sequences,
			start     = start,
			end       = end
		)

		self.sequences = get_encoding(
			sequences     = self.sequences,
			should_encode = onehot,
			expand_dims   = expand_dims,
			encoder       = encode,
			expander      = expand
		)

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
