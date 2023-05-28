from typing import Any
from typing import Dict
from typing import Tuple

import os

from source.python.cnn.cnn_model         import get_model_trainers
from source.python.dataset.dataset_utils import get_dataset
from source.python.io.loader             import load_fasta
from source.python.io.loader             import load_npz
from source.python.raytune.raytune_model import get_dataloaders
from source.python.raytune.raytune_model import get_metrics
from source.python.raytune.raytune_model import get_model
from source.python.raytune.raytune_model import main_loop
from source.python.runtime               import lock_random

def update_core_config (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> Dict[str, Any] :
	"""
	Doc
	"""

	tokens = tune_config['gs/target']
	tokens = tokens.split('-')

	output_target  = tokens[0].lower()
	output_type    = tokens[1].lower()
	output_filter  = None
	output_explode = False

	if len(tokens) >= 3 :
		output_explode = True
		output_filter  = tokens[2].lower()

		if output_filter == 'explode' :
			output_filter = None

	core_config['model/output/target']  = output_target
	core_config['model/output/type']    = output_type
	core_config['model/output/filter']  = output_filter
	core_config['model/output/explode'] = output_explode

	core_config['model/type'] = tune_config['gs/model']

	return core_config

def get_sequences_and_features (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> Tuple[Dict, Any, str] :
	"""
	Doc
	"""

	out = os.path.join(core_config['core/rootdir'], 'output')

	res_nbp04 = os.path.join(out, 'nbp04-feature', tune_config['gs/filter'])
	res_nbp05 = os.path.join(out, 'nbp05-target',  tune_config['gs/filter'])

	sequence = load_fasta(
		filename = os.path.join(res_nbp04, 'sequences-2150-keep.fasta'),
		to_string = True
	)

	feature_base = load_npz(
		filename = os.path.join(res_nbp04, 'features-base-keep.npz')
	)

	return sequence, feature_base, res_nbp05

def main (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	lock_random(seed = core_config['core/random'])

	# For compatibility of core_config in further methods
	core_config = update_core_config(
		tune_config = tune_config,
		core_config = core_config
	)

	sequence, feature_base, directory = get_sequences_and_features(
		tune_config = tune_config,
		core_config = core_config
	)

	dataset = get_dataset(
		config    = core_config,
		sequence  = sequence,
		feature   = feature_base,
		directory = directory,
		cached    = None,
		filename  = 'mapping-grouped-keep.pkl',
		start     = None,
		end       = None
	)[0]

	dataloaders = get_dataloaders(
		core_config = core_config,
		tune_config = tune_config,
		dataset     = dataset
	)

	model = get_model(
		core_config  = core_config,
		tune_config  = tune_config,
		params_share = False
	)

	model_trainers = get_model_trainers(
		model  = model,
		config = tune_config,
		epochs = core_config['model/epochs']
	)

	main_loop(
		core_config  = core_config,
		model_params = {
			'model'     : model,
			'criterion' : model_trainers['criterion'],
			'optimizer' : model_trainers['optimizer'],
			'scheduler' : model_trainers['scheduler'],
			'device'    : core_config['core/device'],
			'verbose'   : False,
			'metrics'   : get_metrics(
				config    = core_config
			),
			'train_dataloader' : dataloaders[0],
			'valid_dataloader' : dataloaders[1],
			'test_dataloader'  : dataloaders[2]
		}
	)
