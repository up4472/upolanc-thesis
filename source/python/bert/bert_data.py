from typing import Callable
from typing import Dict
from typing import List

import numpy
import os

from source.python.dataset.dataset_classes import GeneDataset
from source.python.dataset.dataset_utils   import to_gene_dataset

def create_kmers (sequences : Dict[str, str], features : Dict[str, List], targets : Dict[str, List], generator : Callable, split_size : Dict[str, float], filename : str, random_seed : int = None) -> None :
	"""
	Doc
	"""

	dataset = to_gene_dataset(
		sequences   = sequences,
		features    = features,
		targets     = targets,
		onehot      = False,
		expand_dims = None
	)

	generator = generator(
		dataset     = dataset,
		split_size  = split_size,
		random_seed = random_seed
	)

	indices = next(generator)
	names = ['train', 'valid', 'dev']

	for kmer in range(3, 7) :
		for index, name in zip(indices, names) :
			if index is None : continue

			create_kmer(
				dataset      = dataset,
				indices      = index,
				filename     = filename.format(kmer, name),
				kmer         = kmer,
				write_header = True
			)

def create_kmer (dataset : GeneDataset, indices : numpy.ndarray, filename : str, kmer : int, write_header : bool = True) -> None :
	"""
	Doc
	"""

	item_sep = '\t'
	list_sep = ' '

	float2str = lambda x : str(x)
	array2str = lambda x : list_sep.join([float2str(i) for i in x])

	os.makedirs(os.path.dirname(filename), exist_ok = True)

	with open(filename, mode = 'w') as handle :
		if write_header :
			handle.write('sequence')
			handle.write(item_sep)
			handle.write('label')
			handle.write('\n')

		for index in indices :
			data = dataset[index]

			key      = data[0] # noqa unused
			sequence = data[1]
			feature  = data[2] # noqa unused
			target   = data[3]

			sequence = [sequence[x: x + kmer] for x in range(len(sequence) + 1 - kmer)]
			sequence = list_sep.join(sequence)

			if isinstance(target, list) :
				target = array2str(target)
			elif isinstance(target, numpy.ndarray) :
				target = array2str(target)
			else :
				target = float2str(target)

			handle.write(sequence)
			handle.write(item_sep)
			handle.write(target)
			handle.write('\n')
