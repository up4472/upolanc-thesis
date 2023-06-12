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
from source.python.dataset.dataset_utils          import get_dataset
from source.python.io.loader                      import load_csv
from source.python.io.loader                      import load_faidx
from source.python.io.loader                      import load_json
from source.python.raytune.raytune_model          import get_dataloaders
from source.python.raytune.raytune_model          import get_metrics
from source.python.raytune.raytune_model          import get_model
from source.python.raytune.raytune_model          import main_loop
from source.python.runtime                        import lock_random

CACHE = {
	'nbp04/sequence/lengths' : {
		'prom_utr5' : [int( 5000), int(  0)],
		'prom_full' : [int( 5000), int(  0)],
		'prom'      :  int( 1000),
		'utr5'      :  int(  300),
		'cds'       :  int(10000),
		'utr3'      :  int(  350),
		'term'      :  int(  500),
		'term_full' : [int(    0), int(500)]
	},
	'nbp04/sequence/padding' : {
		'prom_utr5' : 'left',
		'prom_full' : 'left',
		'prom'      : 'left',
		'utr5'      : 'left',
		'cds'       : 'none',
		'utr3'      : 'left',
		'term'      : 'right',
		'term_full' : 'right'
	}
}

def get_anndata (core_config : Dict[str, Any], tune_config : Dict[str, Any]) -> AnnData :
	"""
	Doc
	"""

	cached_anndata = 'nbp02/anndata' in CACHE.keys()

	if not cached_anndata :
		with warnings.catch_warnings() :
			warnings.simplefilter('ignore')

			CACHE['nbp02/anndata'] = create_anndata(
				mat = load_csv(filename = os.path.join(core_config['core/rootdir'], 'output', 'nbp01-filter', core_config['core/subfolder'], 'tissue-tpm.csv')),
				obs = load_csv(filename = os.path.join(core_config['core/rootdir'], 'output', 'nbp01-filter', core_config['core/subfolder'], 'tissue-metadata.csv'))
			)

	return compute_boxcox1p(
		data       = CACHE['nbp02/anndata'],
		store_into = 'boxcox1p',
		layer      = None,
		eps        = 1.0,
		lmbda      = tune_config['boxcox/lambda']
	)[0]

def get_sequences_and_features (core_config : Dict[str, Any], prom_length : int = 1000) -> Tuple[Dict, Dict] :
	"""
	Doc
	"""

	CACHE['nbp04/sequence/lengths']['prom'] = int(prom_length)

	cached_sequence = 'nbp04/sequence/full' in CACHE.keys()
	cached_features = 'nbp04/features/base' in CACHE.keys()

	if not (cached_sequence and cached_features) :
		file1 = os.path.join(core_config['core/rootdir'], 'output', 'nbp01-filter', core_config['core/subfolder'], 'gene-annotation.csv')
		file2 = os.path.join(core_config['core/rootdir'], 'resources', 'genome', 'arabidopsis-r36', 'gene-assembly.fa')
		file3 = os.path.join(core_config['core/rootdir'], 'output', 'nbp01-filter', core_config['core/subfolder'], 'filter.json')

		regions = annotation_to_regions(
			lengths    = CACHE['nbp04/sequence/lengths'],
			verbose    = False,
			annotation = load_csv(
				filename   = file1,
				low_memory = False
			)
		)

		regions['UTR_Min'] = regions[['Start', 'End']].min(axis = 1)
		regions['UTR_Max'] = regions[['Start', 'End']].max(axis = 1)

		regions.drop(columns = ['Start', 'End'])

		regions['CDS_Min'] = regions['CDS'].apply(lambda x : numpy.min(x) - 1)
		regions['CDS_Max'] = regions['CDS'].apply(lambda x : numpy.max(x) - 1)

		regions['Prom_UTR5'] = None

		posx = lambda x : max(x - CACHE['nbp04/sequence/lengths']['prom_utr5'][0], 1)
		posy = lambda x :     x + CACHE['nbp04/sequence/lengths']['prom_utr5'][1]

		negx = lambda x : max(x - CACHE['nbp04/sequence/lengths']['prom_utr5'][1], 1)
		negy = lambda x :     x + CACHE['nbp04/sequence/lengths']['prom_utr5'][0]

		regions.loc[regions['Strand'] == '+', 'Prom_UTR5'] = regions[regions['Strand'] == '+']['CDS_Min'].apply(lambda x : [[posx(x), posy(x)]])
		regions.loc[regions['Strand'] == '-', 'Prom_UTR5'] = regions[regions['Strand'] == '-']['CDS_Max'].apply(lambda x : [[negx(x), negy(x)]])

		sequences, features = regions_to_features(
			dataframe = regions,
			lengths   = CACHE['nbp04/sequence/lengths'],
			verbose   = False,
			faidx     = load_faidx(
				filename = file2
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

		filter_dict = load_json(
			filename = file3,
		)

		sequences = {k : v for k, v in sequences.items() if k in filter_dict['data']['keep_transcript']}
		features  = {k : v for k, v in features.items()  if k in filter_dict['data']['keep_transcript']}

		sequences =  sequences_extend_kvpair(
			sequences = sequences,
			regions   = regions,
			header    = '{} | {} | {}:{}-{} | {}'
		)

		sequences, _ = merge_and_pad_sequences(
			sequences = sequences,
			lengths   = CACHE['nbp04/sequence/lengths'],
			padding   = CACHE['nbp04/sequence/padding'],
		)

		CACHE['nbp04/sequence/full'] = {
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
		CACHE['nbp04/sequence/full'],
		CACHE['nbp04/features/base']
	)

def get_targets (core_config : Dict[str, Any], tune_config : Dict[str, Any], data : AnnData, layer : str = None) -> Dict :
	"""
	Doc
	"""

	with warnings.catch_warnings() :
		warnings.simplefilter('ignore')

		values, order = extract_tpm_multi(
			data      = data,
			layer     = layer,
			verbose   = False,
			groups    = ['Tissue', 'Age', 'Group', 'Perturbation'],
			functions = [
				('max',  lambda x, axis : numpy.nanmax(x, axis = axis)),
				('mean', lambda x, axis : numpy.nanmean(x, axis = axis))
			],
			outlier_filter = 'zscore',
			outlier_params = {
				'factor-zscore' : 3.0,
				'factor-iqr'    : 1.5
			}
		)

	if layer is None : matrix = data.X
	else             : matrix = data.layers[layer]

	for index, transcript in enumerate(data.var.index) :
		values[transcript]['global-mean'] = [numpy.nanmean(matrix[:, index], axis = None)]
		values[transcript]['global-max']  = [numpy.nanmax(matrix[:, index], axis = None)]

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

		keep  = [x for x in keep if x in order[key]]
		index = [order[key].index(x) for x in keep]

		order[key] = keep

		for transcript in values.keys() :
			for group, array in values[transcript].items() :
				if not group.startswith(key.lower()) :
					continue

				values[transcript][group] = [array[x] for x in index]

	labels, bounds = classify_tpm(
		data    = values,
		classes = tune_config['class/bins']
	)

	_, features_grouped, _ = create_mapping(
		values = values,
		labels = labels,
		order  = order
	)

	filter_dict = load_json(
		filename = os.path.join(core_config['core/rootdir'], 'output', 'nbp01-filter', core_config['core/subfolder'], 'filter.json'),
	)

	return {
		key : dataframe[dataframe['Transcript'].isin(filter_dict['data']['keep_transcript'])].copy()
		for key, dataframe in features_grouped.items()
	}

def get_model_params (config : Dict[str, Any]) -> List[Dict[str, Any]] :
	"""
	Doc
	"""

	folder = config['params/filepath']
	params = [{}]

	if config['model/name'].startswith('zrimec') :
		if 'params/zrimec' not in CACHE.keys() :
			filename = os.path.join(folder, 'zrimec.json')

			if os.path.exists(filename) :
				CACHE['params/zrimec'] = load_json(filename = filename)
			else :
				CACHE['params/zrimec'] = None

		params = CACHE['params/zrimec']

	if config['model/name'].startswith('washburn') :
		if 'params/washburn' not in CACHE.keys() :
			filename = os.path.join(folder, 'washburn.json')

			if os.path.exists(filename) :
				CACHE['params/washburn'] = load_json(filename = filename)
			else :
				CACHE['params/washburn'] = None

		params = CACHE['params/washburn']

	return params

def main (tune_config : Dict[str, Any], core_config : Dict[str, Any]) -> None :
	"""
	Doc
	"""

	lock_random(seed = core_config['core/random'])

	data = get_anndata(
		core_config = core_config,
		tune_config = tune_config
	)

	sequence, feature = get_sequences_and_features(
		core_config = core_config
	)

	data = data[:, list(feature.keys())].copy()

	params = get_model_params(
		config = core_config
	)

	if params is not None : core_config['params/tuner'] = params
	else                  : core_config['params/tuner'] = None

	cached = get_targets(
		core_config = core_config,
		tune_config = tune_config,
		data        = data,
		layer       = 'boxcox1p'
	)

	dataset = get_dataset(
		config    = core_config,
		sequence  = sequence,
		feature   = feature,
		directory = core_config['core/outdir'],
		cached    = cached,
		start     = core_config['dataset/sequence/start'],
		end       = core_config['dataset/sequence/end'],
		filename  = 'mapping-grouped-keep.pkl'
	)[0]

	dataloaders = get_dataloaders(
		core_config  = core_config,
		tune_config  = core_config['params/tuner'],
		dataset      = dataset
	)

	model = get_model(
		core_config = core_config,
		tune_config = core_config['params/tuner']
	)

	model_trainers = get_model_trainers(
		model  = model,
		config = core_config['params/tuner'],
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
				config    = core_config,
				n_classes = tune_config['class/bins']
			),
			'train_dataloader' : dataloaders[0],
			'valid_dataloader' : dataloaders[1],
			'test_dataloader'  : dataloaders[2]
		}
	)
