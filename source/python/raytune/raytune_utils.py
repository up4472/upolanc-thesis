from ray.air                  import CheckpointConfig
from ray.air                  import FailureConfig
from ray.air                  import RunConfig
from ray.tune                 import CLIReporter
from ray.tune                 import JupyterNotebookReporter
from ray.tune                 import TuneConfig
from ray.tune                 import Tuner
from ray.tune.schedulers      import ASHAScheduler
from ray.tune.search          import BasicVariantGenerator
from ray.tune.search          import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper         import TimeoutStopper
from typing                   import Any
from typing                   import Callable
from typing                   import Dict
from typing                   import List

import datetime
import ray

VERBOSE_SILENT = 0
VERBOSE_STATUS = 1
VERBOSE_BRIEF  = 2
VERBOSE_DETAIL = 3

def get_search_algorithm (config : Dict[str, Any], metric : str = 'valid_loss', mode : str = 'min', params : List[Dict[str, Any]] = None, algorithm : str = 'hyperopt') -> Any :
	"""
	Doc
	"""

	algorithm = algorithm.lower()

	if algorithm in ['hyperopt', 'hyperopt-search', 'hyperoptsearch'] :
		return ConcurrencyLimiter(
			searcher = HyperOptSearch(
				n_initial_points   = 20,
				points_to_evaluate = params,
				metric             = metric,
				mode               = mode,
				space              = None,
				gamma              = 0.25,
				random_state_seed  = None
			),
			max_concurrent = config['tuner/max_concurrent']
		)

	if algorithm in ['grid', 'grid-search', 'gridsearch'] :
		return BasicVariantGenerator(
			points_to_evaluate   = params,
			max_concurrent       = config['tuner/max_concurrent'],
			constant_grid_search = True,
			random_state         = None
		)

	if algorithm in ['random', 'random-search', 'randomsearch'] :
		return BasicVariantGenerator(
			points_to_evaluate   = params,
			max_concurrent       = config['tuner/max_concurrent'],
			constant_grid_search = True,
			random_state         = None
		)

	raise ValueError()

def create_tune_config (config : Dict[str, Any], metric : str = 'valid_loss', mode : str = 'min', params : List[Dict[str, Any]] = None, algorithm : str = 'hyperopt') -> TuneConfig :
	"""
	Doc
	"""

	tune_searcher = get_search_algorithm(
		config    = config,
		metric    = metric,
		mode      = mode,
		params    = params,
		algorithm = algorithm
	)

	tune_scheduler = ASHAScheduler(
		time_attr        = 'training_iteration',
		max_t            = config['tuner/max_epochs'],
		grace_period     = config['tuner/min_epochs'],
		reduction_factor = 4,
		brackets         = 1
	)

	return TuneConfig(
		metric      = metric,
		mode        = mode,
		num_samples = config['tuner/trials'],
		search_alg  = tune_searcher,
		scheduler   = tune_scheduler,
		trial_name_creator    = lambda x : str(x.trial_id),
		trial_dirname_creator = lambda x : str(x.trial_id)
	)

def create_run_config (config : Dict[str, Any], local_dir : str = None, verbosity : int = 1, task : str = 'model') -> RunConfig :
	"""
	Doc
	"""

	reporter = CLIReporter
	stopper  = None

	if config['tuner/reporter/notebook'] :
		reporter = JupyterNotebookReporter

	pcolumns = None
	mcolumns = None

	if config['model/mode'] == 'regression' :
		mcolumns = [
			'train_loss', 'train_r2', 'train_mae',
			'valid_loss', 'valid_r2', 'valid_mae'
		]

		if   task == 'model'   : pcolumns = ['dataset/batch_size', 'optimizer/name', 'scheduler/name']
		elif task == 'data'    : pcolumns = ['boxcox/lambda']
		elif task == 'feature' : pcolumns = ['dataset/batch_size', 'optimizer/name', 'scheduler/name']

	elif config['model/mode'] == 'classification' :
		mcolumns = [
			'train_loss', 'train_accuracy', 'train_auroc', 'train_f1', 'train_matthews',
			'valid_loss', 'valid_accuracy', 'valid_auroc', 'valid_f1', 'valid_matthews'
		]

		if   task == 'model'   : pcolumns = ['dataset/batch_size', 'optimizer/name', 'scheduler/name']
		elif task == 'data'    : pcolumns = ['boxcox/lambda', 'class/bins']
		elif task == 'feature' : pcolumns = ['dataset/batch_size', 'optimizer/name', 'scheduler/name']

	reporter = reporter(
		max_column_length    = 32,
		max_progress_rows    = 10,
		parameter_columns    = pcolumns,
		metric_columns       = mcolumns,
		max_report_frequency = 60 * config['tuner/reporter/freq']
	)

	failure = FailureConfig(
		max_failures = 0
	)

	checkpoint = CheckpointConfig(
		num_to_keep = None
	)

	if config['tuner/stopper'] :
		stopper = TimeoutStopper(
			timeout = datetime.timedelta(
				days    = config['tuner/stopper/days'],
				hours   = config['tuner/stopper/hours'],
				minutes = config['tuner/stopper/minutes']
			)
		)

	return RunConfig(
		name              = 'raytune',
		local_dir         = local_dir,
		callbacks         = None,
		log_to_file       = True,
		stop              = stopper,
		verbose           = verbosity,
		failure_config    = failure,
		checkpoint_config = checkpoint,
		progress_reporter = reporter
	)

def create_trainable (method : Callable, config : Dict[str, Any], cpu_count : int, gpu_count : int) -> Any :
	"""
	Doc
	"""

	return ray.tune.with_resources(
		trainable = lambda x : method(tune_config = x, core_config = config),
		resources = {
			'cpu' : max(1, cpu_count),
			'gpu' : max(0, gpu_count)
		}
	)

def create_tuner (trainable : Any, tune_config : TuneConfig, run_config : RunConfig, param_space : Dict[str, Any]) -> Tuner :
	"""
	Doc
	"""

	return Tuner(
		trainable   = trainable,
		tune_config = tune_config,
		run_config  = run_config,
		param_space = param_space
	)
