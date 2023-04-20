from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple

import numpy
import os

from source.python.dataset.dataset_classes import GeneDataset
from source.python.dataset.dataset_utils   import to_gene_dataset
from source.python.io.loader               import load_feature_targets

def sequence_to_tokens (sequence : str, kmer : int) -> List[str] :
	"""
	Doc
	"""

	if len(sequence) <  kmer : return []
	if len(sequence) == kmer : return [sequence]

	n = len(sequence) - kmer + 1

	return [sequence[x:x + kmer] for x in range(n)]

def tokens_to_sequence (tokens : List[str]) -> str :
	"""
	Doc
	"""

	if len(tokens) == 0 : return ''
	if len(tokens) <= 1 : return tokens[0]

	last = tokens[-1]
	head = tokens[:-1]

	return ''.join([x[0] for x in head]) + str(last)

def data_prepare (sequences : Dict[str, str], features : Dict[str, Any], directory : str, valid_split : float, test_split : float) -> Generator[Tuple[Dict, str], None, None] :
	"""
	Doc
	"""

	combinations = [
		('global', 'mean', False, None),
		('global', 'max',  False, None),
		('tissue', 'mean', False, None),
		('tissue', 'mean', True,  None),
		('tissue', 'mean', True, 'seedling'),
		('tissue', 'mean', True, 'leaf'),
		('group',  'mean', False, None),
		('group',  'mean', True,  None),
		('group',  'mean', True, 'young_seedling'),
		('group',  'mean', True, 'mature_leaf'),
	]

	for combination in combinations :
		tgroup   = combination[0]
		ttype    = combination[1]
		texplode = combination[2]
		tfilter  = combination[3]

		t01 = '{}-{}'.format(tgroup, ttype)
		t04 = t01

		if texplode :
			t04 = '{}-{}'.format(t04, 'explode' if tfilter is None else tfilter)

		dataframe, target_value, target_order = load_feature_targets(
			group    = t01,
			explode  = bool(texplode),
			filters  = {
				'tissue'       : None,
				'group'        : None,
				'age'          : None,
				'perturbation' : None,
				'global'       : None
			} | {
				tgroup : tfilter
				if tfilter is None
				else [tfilter]
			},
			directory = directory,
			filename  = 'mapping-grouped.pkl'
		)

		if 'Feature' in dataframe.columns :
			feature_extended = {
				key : numpy.concatenate((features[key.split('?')[-1]], value))
				for key, value in dataframe['Feature'].to_dict().items()
			}
		else :
			feature_extended = features

		yield (
			{
				'sequences'  : sequences,
				'features'   : feature_extended,
				'targets'    : target_value,
				'split_size' : {
					'valid' : valid_split,
					'test'  : test_split
				}
			},
			t04
		)

def create_kmers (data : Dict[str, Dict], generator : Callable, filename : str, max_tokens : int = None, random_seed : int = None) -> None :
	"""
	Doc
	"""

	sequences  = data['sequences']
	features   = data['features']
	targets    = data['targets']
	split_size = data['split_size']

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

	for kmer in [3, 6] :
		for index, name in zip(indices, names) :
			if index is None : continue

			create_kmer(
				dataset      = dataset,
				indices      = index,
				filename     = filename.format(kmer, name),
				kmer         = kmer,
				max_tokens   = max_tokens
			)

def create_kmer (dataset : GeneDataset, indices : numpy.ndarray, filename : str, kmer : int, max_tokens : int = None) -> None :
	"""
	Doc
	"""

	item_sep = '\t'
	list_sep = ' '
	line_sep = '\n'

	float2str = lambda x : str(x)
	array2str = lambda x : list_sep.join([float2str(i) for i in x])

	os.makedirs(os.path.dirname(filename), exist_ok = True)

	with open(filename, mode = 'w') as handle :
		handle.write('sequence')
		handle.write(item_sep)
		handle.write('label')
		handle.write(item_sep)
		handle.write('feature')
		handle.write(item_sep)
		handle.write('key')
		handle.write(line_sep)

		for index in indices :
			data = dataset[index]

			key      = data[0]
			sequence = data[1]
			feature  = data[2]
			target   = data[3]

			sequence = sequence_to_tokens(
				sequence = sequence,
				kmer     = kmer
			)

			if max_tokens is not None and max_tokens < len(sequence) :
				if max_tokens <= 0 : sequence = sequence[max_tokens:]
				if max_tokens >= 0 : sequence = sequence[:max_tokens]

			sequence = list_sep.join(sequence)

			if   isinstance(target, list)           : target = array2str(target)
			elif isinstance(target, numpy.ndarray)  : target = array2str(target)
			else                                    : target = float2str(target)

			if   isinstance(feature, list)          : feature = array2str(feature)
			elif isinstance(feature, numpy.ndarray) : feature = array2str(feature)
			else                                    : feature = float2str(feature)

			handle.write(sequence)
			handle.write(item_sep)
			handle.write(target)
			handle.write(item_sep)
			handle.write(feature)
			handle.write(item_sep)
			handle.write(key)
			handle.write(line_sep)
