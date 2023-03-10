from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from typing                  import Any
from typing                  import Dict

import numpy

from source.python.dataset.dataset_classes import GeneDataset

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
