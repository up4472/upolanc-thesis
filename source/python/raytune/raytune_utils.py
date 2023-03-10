from ray.air                  import CheckpointConfig
from ray.air                  import FailureConfig
from ray.air                  import RunConfig
from ray.tune                 import CLIReporter
from ray.tune                 import JupyterNotebookReporter
from ray.tune                 import TuneConfig
from ray.tune                 import Tuner
from ray.tune.schedulers      import ASHAScheduler
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

def create_tune_config (config : Dict[str, Any], params : List[Dict[str, Any]] = None) -> TuneConfig :
	"""
	Doc
	"""

	tune_searcher = HyperOptSearch(
		points_to_evaluate = params,
		metric             = 'valid_loss',
		mode               = 'min'
	)

	tune_searcher = ConcurrencyLimiter(
		searcher       = tune_searcher,
		max_concurrent = config['tuner/max_concurrent']
	)

	tune_scheduler = ASHAScheduler(
		time_attr        = 'training_iteration',
		max_t            = config['tuner/max_epochs'],
		grace_period     = config['tuner/min_epochs'],
		reduction_factor = 4,
		brackets         = 1
	)

	return TuneConfig(
		metric      = 'valid_loss',
		mode        = 'min',
		num_samples = config['tuner/trials'],
		search_alg  = tune_searcher,
		scheduler   = tune_scheduler,
		trial_name_creator    = lambda x : str(x.trial_id),
		trial_dirname_creator = lambda x : str(x.trial_id)
	)

def create_run_config (config : Dict[str, Any], local_dir : str = None, verbosity : int = 1) -> RunConfig :
	"""
	Doc
	"""

	reporter = CLIReporter
	stopper  = None

	if config['tuner/reporter/notebook'] :
		reporter = JupyterNotebookReporter

	reporter = reporter(
		max_column_length    = 32,
		max_progress_rows    = 10,
		parameter_columns    = ['dataset/batch_size', 'optimizer/name', 'optimizer/lr', 'scheduler/name', 'model/dropout'],
		metric_columns       = ['valid_loss', 'valid_r2', 'train_loss'],
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
