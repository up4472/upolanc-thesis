import os.path
from typing import Callable
from typing import Dict
from typing import List

import textwrap
import numpy

from source.python.cnn.dataset import GeneDataset
from source.python.cnn.dataset import to_dataset

def prepare_dnabert_data (sequences : Dict[str, str], features : Dict[str, List], targets : Dict[str, List], generator : Callable, split_size : Dict[str, float], filename : str, write_header : bool = True, remainder : str = 'keep', random_seed : int = None) -> None :
	"""
	Doc
	"""

	dataset = to_dataset(
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
	kmers = {3, 4, 5, 6}

	for kmer in kmers :
		for index, name in zip(indices, names) :
			if index is None : continue

			write_dnabert_data(
				dataset      = dataset,
				indices      = index,
				filename     = filename.format(kmer, name),
				kmer         = kmer,
				remainder    = remainder,
				write_header = write_header
			)

def write_dnabert_data (dataset : GeneDataset, indices : numpy.ndarray, filename : str, kmer : int, remainder : str = 'keep', write_header : bool = True) -> None :
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

			kmers = list()

			for item in textwrap.wrap(sequence, width = kmer) :
				if len(item) < kmer :
					if   remainder == 'keep' :
						kmers.append(item)
					elif remainder == 'drop' :
						continue
					elif remainder == 'fill' :
						kmers.append(item.ljust(kmer, '-'))
					else :
						continue
				else :
					kmers.append(item)

			sequence = list_sep.join(kmers)

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
