from anndata import AnnData
from typing  import Any
from typing  import Dict
from typing  import List
from typing  import Tuple

import numpy
import os
import warnings

from source.python.cnn.cnn_model                  import get_model_trainers
from source.python.data.feature.feature_anndata   import compute_boxcox1p
from source.python.data.feature.feature_anndata   import create_anndata
from source.python.data.feature.feature_extractor import annotation_to_regions
from source.python.data.feature.feature_extractor import merge_and_pad_sequences
from source.python.data.feature.feature_extractor import regions_to_features
from source.python.data.feature.feature_extractor import sequences_extend_kvpair
from source.python.data.feature.feature_target    import classify_tpm
from source.python.data.feature.feature_target    import create_mapping
from source.python.data.feature.feature_target    import extract_tpm_multi
from source.python.dataset.dataset_classes        import GeneDataset
from source.python.dataset.dataset_utils          import to_gene_dataset
from source.python.io.loader                      import load_csv
from source.python.io.loader                      import load_faidx
from source.python.io.loader                      import load_feature_targets
from source.python.io.loader                      import load_json
from source.python.raytune.raytune_model          import get_dataloaders
from source.python.raytune.raytune_model          import get_metrics
from source.python.raytune.raytune_model          import get_model
from source.python.raytune.raytune_model          import regression_loop
from source.python.runtime                        import lock_random

CACHE = {
	'nbp04/sequence/lengths' : {
		'prom_full' : [int(1000), int(500)],
		'prom'      :  int(1000),
		'utr5'      :  int( 300),
		'cds'       :  int(9999),
		'utr3'      :  int( 350),
		'term'      :  int( 500),
		'term_full' : [int( 500), int(500)]
	},
	'nbp04/sequence/padding' : {
		'prom_full' : 'left',
		'prom'      : 'left',
		'utr5'      : 'left',
		'cds'       : 'none',
		'utr3'      : 'left',
		'term'      : 'right',
		'term_full' : 'right'
	}
}

def get_anndata (params : Dict[str, Any], config : Dict[str, Any]) -> AnnData :
	"""
	Doc
	"""

	if 'nbp02/anndata' not in CACHE.keys() :
		with warnings.catch_warnings() :
			warnings.simplefilter('ignore')

			CACHE['nbp02/anndata'] = create_anndata(
				mat = load_csv(filename = os.path.join(config['core/rootdir'], 'output', 'nbp01-analysis', 'tissue-tpm.csv')),
				obs = load_csv(filename = os.path.join(config['core/rootdir'], 'output', 'nbp01-analysis', 'tissue-metadata.csv'))
			)

	return compute_boxcox1p(
		data       = CACHE['nbp02/anndata'],
		store_into = 'boxcox1p',
		layer      = None,
		eps        = 1.0,
		lmbda      = params['boxcox/lambda']
	)[0]

def get_sequences_and_features (config : Dict[str, Any]) -> Tuple[Dict, Dict] :
	"""
	Doc
	"""

	if 'nbp04/features/bp2150' not in CACHE.keys() or 'nbp04/features/base' not in CACHE.keys() :
		regions = annotation_to_regions(
			lengths    = CACHE['nbp04/sequence/lengths'],
			verbose    = False,
			annotation = load_csv(
				filename   = os.path.join(config['core/rootdir'], 'output', 'nbp01-analysis', 'gene-annotation.csv'),
				low_memory = False
			)
		)

		sequences, features = regions_to_features(
			dataframe = regions,
			lengths   = CACHE['nbp04/sequence/lengths'],
			verbose   = False,
			faidx     = load_faidx(
				filename = os.path.join(config['core/rootdir'], 'resources', 'genome', 'arabidopsis-r36', 'gene-assembly.fa')
			)
		)

		sequences = sequences.copy()
		sequences = sequences.set_index('Transcript', drop = False)
		sequences = sequences.rename_axis(None, axis = 'index') # noqa :: unexpected type
		sequences = sequences.to_dict('index')

		features = features.copy()
		features = features.set_index('Transcript', drop = False)
		features = features.rename_axis(None, axis = 'index') # noqa :: unexpected type
		features = features.to_dict('index')

		sequences =  sequences_extend_kvpair(
			sequences = sequences,
			regions   = regions,
			header    = '{} | {} | {}:{}-{} | {}'
		)

		sequences = merge_and_pad_sequences(
			sequences = sequences,
			lengths   = CACHE['nbp04/sequence/lengths'],
			padding   = CACHE['nbp04/sequence/padding'],
		)

		CACHE['nbp04/features/bp2150'] = {
			k.split()[0] : v
			for k, v in sequences.items()
		}

		features_frequency = {
			key : numpy.array(value['Frequency'])
			for key, value in features.items()
		}

		features_stability = {
			key : numpy.array(value['Stability'])
			for key, value in features.items()
		}

		features_base = dict()

		for key in features_frequency.keys() :
			freq = features_frequency[key]
			stab = features_stability[key]

			features_base[key] = numpy.concatenate((freq, stab), axis = 0)

		CACHE['nbp04/features/base'] = features_base

	return (
		CACHE['nbp04/features/bp2150'],
		CACHE['nbp04/features/base']
	)

def get_targets (data : AnnData, layer : str = None) -> Dict :
	"""
	Doc
	"""

	values, order = extract_tpm_multi(
		data      = data,
		layer     = layer,
		groups    = ['Tissue', 'Age', 'Group', 'Perturbation'],
		functions = [
			('max',  lambda x : numpy.max(x, axis = 0)),
			('mean', lambda x : numpy.mean(x, axis = 0))
		]
	)

	if layer is None :
		matrix = data.X
	else :
		matrix = data.layers[layer]

	for index, transcript in enumerate(data.var.index) :
		values[transcript]['global-mean'] = [matrix[:, index].mean()]
		values[transcript]['global-max']  = [matrix[:, index].max()]

	order['global'] = ['global']

	filters = {
		'tissue'       : None,
		'age'          : None,
		'group'        : ['mature_leaf', 'mature_flower', 'mature_root', 'mature_seed', 'young_seedling'],
		'perturbation' : None
	}

	for key, keep in filters.items() :
		if keep is None :
			continue

		index = [order[key].index(x) for x in keep]
		order[key] = keep

		for transcript in values.keys() :
			for group, array in values[transcript].items() :
				if not group.startswith(key.lower()) :
					continue

				values[transcript][group] = [array[x] for x in index]

	labels, bounds = classify_tpm(
		data    = values,
		classes = 3
	)

	return create_mapping(
		values = values,
		labels = labels,
		order  = order
	)[1]

def get_dataset (config : Dict[str, Any], bp2150 : Dict[str, Any], feature : Dict[str, Any], cached : Dict[str, Any] = None) -> GeneDataset :
	"""
	Doc
	"""

	target_group   = config['model/output/target']
	target_type    = config['model/output/type']
	target_filter  = config['model/output/filter']
	target_explode = config['model/output/explode']

	if config['model/type'].endswith('r') :
		mode = 'regression'
	else :
		mode = 'classification'

	filters = {
		'tissue'       : None,
		'group'        : None,
		'age'          : None,
		'perturbation' : None,
		'global'       : None
	} | {
		target_group : target_filter
		if target_filter is None
		else [target_filter]
	}

	dataframe, target_value, target_order = load_feature_targets(
		group     = '{}-{}'.format(target_group, target_type),
		explode   = target_explode,
		filters   = filters,
		directory = config['core/outdir'],
		filename  = 'mapping-grouped.pkl',
		mode      = mode,
		cached    = cached
	)

	if 'Feature' in dataframe.columns :
		feature = {
			key : numpy.concatenate((feature[key.split('?')[-1]], value))
			for key, value in dataframe['Feature'].to_dict().items()
		}

	if mode == 'regression' :
		config['model/output/size']    = len(target_order)
		config['model/input/features'] = len(list(feature.values())[0])
	else :
		config['model/output/size']    = len(numpy.unique(numpy.array([x for x in dataframe['TPM_Label']]).flatten()))
		config['model/output/heads']   = len(target_order)
		config['model/input/features'] = len(list(feature.values())[0])

	return to_gene_dataset(
		sequences   = bp2150,
		features    = feature,
		targets     = target_value,
		expand_dims = config['dataset/expanddim'],
		groups      = None
	)

def get_model_params (config : Dict[str, Any]) -> List[Dict[str, Any]] :
	"""
	Doc
	"""

	folder = config['params/filepath']
	params = [{}]

	if config['model/type'].startswith('zrimec2020r') :
		if 'params/zrimec2020' not in CACHE.keys() :
			filename = os.path.join(folder, 'zrimec2020.json')

			if os.path.exists(filename) :
				CACHE['params/zrimec2020'] = load_json(filename = filename)

		params = CACHE['params/zrimec2020']

	if config['model/type'].startswith('washburn2019r') :
		if 'params/washburn2019' not in CACHE.keys() :
			filename = os.path.join(folder, 'washburn2019.json')

			if os.path.exists(filename) :
				CACHE['params/washburn2019'] = load_json(filename = filename)

		params = CACHE['params/washburn2019']

	return params

def main (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	lock_random(seed = core_config['core/random'])

	data            = get_anndata(params = tune_config, config = core_config)
	bp2150, feature = get_sequences_and_features(config = core_config)

	data = data[:, list(feature.keys())].copy()

	core_config['params/tuner'] = get_model_params(
		config = core_config
	)[0]

	dataset = get_dataset(
		config  = core_config,
		bp2150  = bp2150,
		feature = feature,
		cached  = get_targets(
			data  = data,
			layer = 'boxcox1p'
		)
	)

	dataloaders = get_dataloaders(
		params  = core_config['params/tuner'],
		config  = core_config,
		dataset = dataset
	)

	model = get_model(
		params = core_config['params/tuner'],
		config = core_config
	)

	model_trainers = get_model_trainers(
		model  = model,
		config = core_config['params/tuner'],
		epochs = core_config['model/epochs']
	)

	regression_loop(
		config       = core_config,
		model_params = {
			'model'     : model,
			'criterion' : model_trainers['criterion'],
			'optimizer' : model_trainers['optimizer'],
			'scheduler' : model_trainers['scheduler'],
			'device'    : core_config['core/device'],
			'verbose'   : False,
			'metrics'   : get_metrics(config = core_config),
			'train_dataloader' : dataloaders[0],
			'valid_dataloader' : dataloaders[1],
			'test_dataloader'  : dataloaders[2]
		}
	)
