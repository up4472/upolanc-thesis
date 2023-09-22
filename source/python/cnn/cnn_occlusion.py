from typing           import Any
from typing           import Dict
from typing           import Tuple
from torch.nn         import Module
from torch.utils.data import DataLoader

import os
import numpy
import torch

from source.python.cnn.cnn_model           import get_criterion
from source.python.cnn.cnn_model           import get_model_trainers
from source.python.cnn.models              import Washburn2019c
from source.python.cnn.models              import Washburn2019r
from source.python.cnn.models              import Zrimec2020c
from source.python.cnn.models              import Zrimec2020r
from source.python.dataset.dataset_classes import GeneDataset
from source.python.dataset.dataset_utils   import get_dataset
from source.python.dataset.dataset_utils   import to_dataloaders
from source.python.io.loader               import load_fasta
from source.python.io.loader               import load_json
from source.python.io.loader               import load_npz

def compute_relevance (base : float, value : float) -> float :
	"""
	Doc
	"""

	if base < 0 :
		print('The value of [base] is negative.')
		return numpy.nan

	if base < 1e-7 :
		print('The value of [base] is zero or very close.')
		return numpy.nan

	return (base - value) / base

def select_only_evaluation_transcripts (directory : str, report : str) -> Tuple[Dict, Dict] :
	"""
	Doc
	"""

	report = load_json(report)
	report = report['eval']

	transcripts = report['keys']
	transcripts = [x.split('?')[1] for x in transcripts]
	transcripts = list(set(transcripts))

	sequences = os.path.join(directory, 'sequences-2150-keep.fasta')
	sequences = load_fasta(sequences, to_string = True)
	sequences = {k : v for k, v in sequences.items() if k in transcripts}

	features = os.path.join(directory, 'features-base-keep.npz')
	features = load_npz(features)
	features = {k : v for k, v in features.items() if k in transcripts}

	print('Evaluation Transcripts : {}'.format(len(transcripts)))
	print('Evaluation Sequences   : {}'.format(len(sequences)))
	print('Evaluation Features    : {}'.format(len(features)))
	print()

	return sequences, features

def to_dataset (config : Dict[str, Any], directory : str, sequences : Dict[str, str], features : Dict[str, Any]) -> Tuple[GeneDataset, Dict] :
	"""
	Doc
	"""

	dataset, dataframe, target_value, target_order = get_dataset(
		config    = config,
		sequence  = sequences,
		feature   = features,
		directory = directory,
		cached    = None,
		start     = None,
		end       =  None,
		filename  = 'mapping-grouped-keep.pkl'
	)

	print('Feature Size : {}'.format(config['model/input/features']))
	print('Target Size  : {}'.format(config['model/output/size']))
	print('Target Heads : {}'.format(config['model/output/heads']))
	print()

	config['model/fc3/features'] = config['model/output/size']
	config['model/fc3/heads']    = config['model/output/heads']

	return dataset, config

def load_pretrained_model (config : Dict[str, Any], device : Any, path : str, dataloader : DataLoader = None) -> Tuple[Module, Dict] :
	"""
	Doc
	"""

	pretrained = torch.load(path)

	if config['model/arch'] == 'zrimec' :
		if   config['model/mode'] == 'regression'     : model = Zrimec2020r(params = config)
		elif config['model/mode'] == 'classification' : model = Zrimec2020c(params = config, binary = False)
		else : raise ValueError()

		model.load_state_dict(pretrained['models'])

		print(
			model.summary(
				batch_size  = config['dataset/batch/train'],
				in_height   = config['model/input/height'],
				in_width    = config['model/input/width'],
				in_features = config['model/input/features'],
			)
		)

	elif config['model/arch'] == 'washburn' :
		if   config['model/mode'] == 'regression'     : model = Washburn2019r(params = config)
		elif config['model/mode'] == 'classification' : model = Washburn2019c(params = config, binary = False)
		else : raise ValueError()

		model.load_state_dict(pretrained['models'])

		print(model.summary(
			batch_size  = config['dataset/batch/train'],
			in_channels = config['model/input/channels'],
			in_height   = config['model/input/height'],
			in_width    = config['model/input/width'],
			in_features = config['model/input/features'],
		))

	model = model.double() # noqa

	if config['model/mode'] == 'regression' :
		metrics = {
			'mse'   : get_criterion(reduction = 'none', weights = None, query = 'mse'),
			'mae'   : get_criterion(reduction = 'none', weights = None, query = 'mae'),
			'smae'  : get_criterion(reduction = 'none', weights = None, query = 'smae'),
			'mape'  : get_criterion(reduction = 'none', weights = None, query = 'mape',  output_size = config['model/output/size']),
			'wmape' : get_criterion(reduction = 'none', weights = None, query = 'wmape', output_size = config['model/output/size']),
			'r2'    : get_criterion(reduction = 'none', weights = None, query = 'r2',    output_size = config['model/output/size']),
		}

	if config['model/mode'] == 'classification' :
		metrics = {
			'entropy'   : get_criterion(reduction = 'none', weights = None, query = 'entropy'),
			'accuracy'  : get_criterion(reduction = 'none', weights = None, query = 'accuracy',  task = 'multiclass', n_classes = config['model/output/size']),
			'auroc'     : get_criterion(reduction = 'none', weights = None, query = 'auroc',     task = 'multiclass', n_classes = config['model/output/size']),
			'confusion' : get_criterion(reduction = 'none', weights = None, query = 'confusion', task = 'multiclass', n_classes = config['model/output/size']),
			'f1'        : get_criterion(reduction = 'none', weights = None, query = 'f1',        task = 'multiclass', n_classes = config['model/output/size']),
			'jaccardi'  : get_criterion(reduction = 'none', weights = None, query = 'jaccardi',  task = 'multiclass', n_classes = config['model/output/size']),
			'matthews'  : get_criterion(reduction = 'none', weights = None, query = 'matthews',  task = 'multiclass', n_classes = config['model/output/size'])
		}

	metrics = {
		k : v.to(device)
		for k, v in metrics.items() # noqa
	}

	model_trainers = get_model_trainers(
		model  = model,
		config = config | {
			'criterion/threshold' : 0.20
		},
		epochs = config['model/epochs']
	)

	model_trainers['criterion'] = pretrained['criterion']
	model_trainers['optimizer'].load_state_dict(pretrained['optimizer'])

	params = {
		'model'            : model,
		'savebest'         : None,
		'savelast'         : None,
		'savetime'         : None,
		'epochs'           : config['model/epochs'],
		'criterion'        : model_trainers['criterion'],
		'optimizer'        : model_trainers['optimizer'],
		'scheduler'        : model_trainers['scheduler'],
		'device'           : device,
		'verbose'          : config['core/verbose'],
		'metrics'          : metrics,
		'train_dataloader' : None,
		'valid_dataloader' : None,
		'test_dataloader'  : dataloader
	}

	return model, params

def create_dataloader_without_occlusion (config : Dict[str, Any], dataset : GeneDataset) -> DataLoader :
	"""
	Doc
	"""

	dataset.set_transform(
		transform = None
	)

	return to_dataloaders(
		dataset     = dataset,
		generator   = config['dataset/split/generator'],
		random_seed = config['core/random'],
		split_size  = {
			'valid' : 0.0,
			'test'  : 0.0
		},
		batch_size  = {
			'train' : config['dataset/batch/test'],
			'valid' : config['dataset/batch/valid'],
			'test'  : config['dataset/batch/train']
		}
	)[0]

def create_dataloader_with_occlusion (config : Dict[str, Any], dataset : GeneDataset, start : int = 0, end : int = 10, method : str = 'zero') -> DataLoader :
	"""
	Doc
	"""

	method = method.lower()

	if   method == 'zero'    : transform = lambda x : occlusion_zero   (sample = x, scope = (start, end))
	elif method == 'shuffle' : transform = lambda x : occlusion_shuffle(sample = x, scope = (start, end))
	elif method == 'random'  : transform = lambda x : occlusion_random (sample = x, scope = (start, end))
	else : raise ValueError()

	dataset.set_transform(
		transform = transform
	)

	return to_dataloaders(
		dataset     = dataset,
		generator   = config['dataset/split/generator'],
		random_seed = config['core/random'],
		split_size  = {
			'valid' : 0.0,
			'test'  : 0.0
		},
		batch_size  = {
			'train' : config['dataset/batch/test'],
			'valid' : config['dataset/batch/valid'],
			'test'  : config['dataset/batch/train']
		}
	)[0]

def occlusion_zero (sample : Tuple[Any, Any, Any, Any], scope : Tuple[int, int]) -> Tuple[Any, Any, Any, Any] :
	"""
	Doc
	"""

	matrix = sample[1].copy()

	if   numpy.ndim(matrix) == 2 : matrix[   :, scope[0]:scope[1]] = 0
	elif numpy.ndim(matrix) == 3 : matrix[0, :, scope[0]:scope[1]] = 0
	else : raise ValueError()

	return sample[0], matrix, sample[2], sample[3]

def occlusion_shuffle (sample : Tuple[Any, Any, Any, Any], scope : Tuple[int, int]) -> Tuple[Any, Any, Any, Any] :
	"""
	Doc
	"""

	matrix = sample[1].copy()

	if   numpy.ndim(matrix) == 2 : segment = matrix[   :, scope[0]:scope[1]]
	elif numpy.ndim(matrix) == 3 : segment = matrix[0, :, scope[0]:scope[1]]
	else : raise ValueError()

	numpy.random.shuffle(segment)

	if   numpy.ndim(matrix) == 2 : matrix[   :, scope[0]:scope[1]] = segment
	elif numpy.ndim(matrix) == 3 : matrix[0, :, scope[0]:scope[1]]  = segment
	else : raise ValueError()

	return sample

def occlusion_random (sample : Tuple[Any, Any, Any, Any], scope : Tuple[int, int]) -> Tuple[Any, Any, Any, Any] :
	"""
	Doc
	"""

	matrix = sample[1].copy()

	segment = numpy.eye(4)
	segment = segment[numpy.random.choice(segment.shape[0], size = scope[1] - scope[0])]

	if   numpy.ndim(matrix) == 2 : matrix[   :, scope[0]:scope[1]] = segment
	elif numpy.ndim(matrix) == 3 : matrix[0, :, scope[0]:scope[1]] = segment
	else : raise ValueError()

	return sample
