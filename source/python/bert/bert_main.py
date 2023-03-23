from transformers import WEIGHTS_NAME # noqa F821 :: unresolved reference :: added at runtime

from typing   import Dict
from typing   import Tuple
from torch.nn import Module
from typing   import Any
from typing   import Optional
from copy     import deepcopy

import glob
import logging
import numpy
import os
import random
import torch.distributed

from source.python.bert.bert_cache      import load_and_cache_examples
from source.python.bert.bert_checkpoint import sort_chekpoints
from source.python.bert.bert_constants  import MODELS
from source.python.bert.bert_constants  import MODES
from source.python.bert.bert_constants  import PROCESSORS
from source.python.bert.bert_finetune   import evaluate
from source.python.bert.bert_finetune   import predict
from source.python.bert.bert_finetune   import train
from source.python.bert.bert_finetune   import visualize
from source.python.bert.bert_metrics    import compute_metrics

def bert_init_args (args : Any, logger : Optional[Any]) -> Any :
	"""
	Doc
	"""

	if args.should_continue :
		sorted_checkpoints = sort_chekpoints(args)

		if len(sorted_checkpoints) == 0 :
			raise ValueError('Used --should_continue but no checkpoint was found in --output_dir.')
		else :
			args.model_name_or_path = sorted_checkpoints[-1]

	if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir :
		raise ValueError('Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.'.format(args.output_dir))

	if args.local_rank == -1 or args.no_cuda :
		device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
		args.n_gpu = torch.cuda.device_count()
	else :
		torch.cuda.set_device(args.local_rank)
		device = torch.device('cuda', args.local_rank)
		torch.distributed.init_process_group(backend = 'nccl')
		args.n_gpu = 1

	args.device = device

	logging.basicConfig(
		format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
		datefmt = '%m/%d/%Y %H:%M:%S',
		level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)

	if logger is not None :
		logger.warning(
			'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
			args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
		)

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	args.eval_batch_size  = args.per_gpu_eval_batch_size  * max(1, args.n_gpu)
	args.pred_batch_size  = args.per_gpu_pred_batch_size  * max(1, args.n_gpu)

	args.task_name  = args.task_name.lower()
	args.model_type = args.model_type.lower()

	args.output_mode = MODES[args.task_name]

	return args

def bert_init_classes (args : Any, logger : Optional[Any], use_features : bool = False, num_features : int = 72) -> Dict[str, Any] :
	"""
	Doc
	"""

	if args.task_name not in PROCESSORS :
		raise ValueError('Task not found: %s' % args.task_name)

	processor  = PROCESSORS[args.task_name]()
	labels     = processor.get_labels()

	if logger is not None :
		logger.info('Using task        : %s', args.task_name)
		logger.info('Using processor   : %s', type(processor))
		logger.info('Using output_mode : %s', args.output_mode)

	if args.local_rank not in [-1, 0] :
		torch.distributed.barrier()

	config_cls    = MODELS[args.model_type][0]
	model_cls     = MODELS[args.model_type][1]
	tokenizer_cls = MODELS[args.model_type][2]

	if logger is not None :
		logger.info('Using model_type  : %s', args.model_type)
		logger.info('Using config      : %s', config_cls.__name__)
		logger.info('Using model       : %s', model_cls.__name__)
		logger.info('Using tokenizer   : %s', tokenizer_cls.__name__)

	model     = None
	tokenizer = None
	config    = None

	if not args.do_visualize and not args.do_ensemble_pred :
		if args.cache_dir      : cache_dir      = args.cache_dir
		else                   : cache_dir      = None
		if args.tokenizer_name : tokenizer_name = args.tokenizer_name
		else                   : tokenizer_name = args.model_name_or_path
		if args.config_name    : config_name    = args.config_name
		else                   : config_name    = args.model_name_or_path

		config = config_cls.from_pretrained(
			config_name,
			num_labels      = len(labels),
			finetuning_task = args.task_name,
			cache_dir       = cache_dir
		)

		config.use_features                 = use_features
		config.num_features                 = num_features
		config.hidden_dropout_prob          = args.hidden_dropout_prob
		config.attention_probs_dropout_prob = args.attention_probs_dropout_prob

		if args.model_type in ['dnalong', 'dnalongcat'] :
			assert args.max_seq_length % 512 == 0

		config.split         = int(args.max_seq_length / 512)
		config.rnn           = args.rnn
		config.num_rnn_layer = args.num_rnn_layer
		config.rnn_dropout   = args.rnn_dropout
		config.rnn_hidden    = args.rnn_hidden

		tokenizer = tokenizer_cls.from_pretrained(
			tokenizer_name,
			do_lower_case = args.do_lower_case,
			cache_dir     = cache_dir
		)

		model = model_cls.from_pretrained(
			args.model_name_or_path,
			from_tf   = bool(".ckpt" in args.model_name_or_path),
			config    = config,
			cache_dir = cache_dir
		)

		if args.local_rank == 0 :
			torch.distributed.barrier()

		model.to(args.device)

		if logger is not None :
			logger.info('Finish loading model')
			logger.info('Training/evaluation parameters %s', args)

	return {
		'model'         : model,
		'tokenizer'     : tokenizer,
		'config'        : config,
		'model_cls'     : model_cls,
		'tokenizer_cls' : tokenizer_cls,
		'config_cls'    : config_cls,
		'labels'        : labels,
		'num_labels'    : len(labels)
	}

def bert_train (args : Any, model : Module, tokenizer : Any, model_cls : Any, tokenizer_cls : Any, logger : Optional[Any], use_features : bool = False) -> Tuple[Module, Any] :
	"""
	Doc
	"""

	if not args.do_train :
		return model, tokenizer

	train_dataset = load_and_cache_examples(
		args            = args,
		task            = args.task_name,
		tokenizer       = tokenizer,
		should_evaluate = False,
		use_features    = use_features
	)

	global_step, training_loss = train(
		args          = args,
		train_dataset = train_dataset,
		model         = model,
		tokenizer     = tokenizer,
		use_features  = use_features
	)

	if logger is not None :
		logger.info('global_step = %s, average loss = %s', global_step, training_loss)

	if args.task_name != 'dna690' and (args.local_rank == -1 or torch.distributed.get_rank() == 0) :
		if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0] :
			os.makedirs(args.output_dir)

		if hasattr(model, 'module') :
			model_to_save = model.module
		else :
			model_to_save = model

		model_to_save.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)
		torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

		model     = model_cls.from_pretrained(args.output_dir)
		tokenizer = tokenizer_cls.from_pretrained(args.output_dir)

		model.to(args.device)

	return model, tokenizer

def bert_evaluate (args : Any, model_cls : Any, tokenizer_cls : Any, logger : Optional[Any], use_features : bool = False) -> Tuple[Any, Any, Dict] :
	"""
	Doc
	"""

	results = dict()

	model     = None
	tokenizer = None

	if not (args.do_eval and args.local_rank in [-1, 0]) :
		return model, tokenizer, results

	tokenizer = tokenizer_cls.from_pretrained(
		args.output_dir,
		do_lower_case = args.do_lower_case
	)

	if args.eval_all_checkpoints :
		checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive = True)))
	else :
		checkpoints = [args.output_dir]

	if logger is not None :
		logger.info('Evaluate the following checkpoints: %s', checkpoints)

	for checkpoint in checkpoints :
		global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ''
		prefix      = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ''

		model = model_cls.from_pretrained(checkpoint)
		model.to(args.device)

		result = evaluate(
			args         = args,
			model        = model,
			tokenizer    = tokenizer,
			prefix       = prefix,
			use_features = use_features
		)

		result = dict((key + '_{}'.format(global_step), value) for key, value in result.items())
		results.update(result)

	return model, tokenizer, results

def bert_predict (args : Any, model_cls : Any, tokenizer_cls : Any, logger : Optional[Any], use_features : bool = False) -> Tuple[Any, Any, Dict] :
	"""
	Doc
	"""

	model     = None
	tokenizer = None

	if not (args.do_predict and args.local_rank in [-1, 0]) :
		return model, tokenizer, dict()

	tokenizer = tokenizer_cls.from_pretrained(
		args.output_dir,
		do_lower_case = args.do_lower_case
	)

	if logger is not None :
		logger.info('Predict using the following checkpoint: %s', args.output_dir)

	model = model_cls.from_pretrained(args.output_dir)
	model.to(args.device)

	results = predict(
		args         = args,
		model        = model,
		tokenizer    = tokenizer,
		prefix       = '',
		use_features = use_features
	)

	return model, tokenizer, results

def bert_visualize (args : Any, model_cls : Any, tokenizer_cls : Any, config_cls : Any, num_labels : int, logger : Optional[Any], use_features : bool = False) -> None :
	"""
	Doc
	"""

	if not (args.do_visualize and args.local_rank in [-1, 0]) :
		return

	if args.visualize_models :
		models = [args.visualize_models]
	else :
		models = [3, 4, 5, 6]

	scores = None
	probs  = None

	for kmer in models :
		output_dir = args.output_dir.replace('/690', '/690/' + str(kmer))

		tokenizer = tokenizer_cls.from_pretrained(
			'dna' + str(kmer),
			do_lower_case = args.do_lower_case,
			cache_dir     = args.cache_dir if args.cache_dir else None,
		)

		checkpoint = output_dir

		if logger is not None :
			logger.info('Calculate attention score using the following checkpoint: %s', checkpoint)

		if checkpoint.find('checkpoint') != -1 :
			prefix = checkpoint.split('/')[-1]
		else :
			prefix = ''

		config = config_cls.from_pretrained(
			output_dir,
			num_labels = num_labels,
			finetuning_task = args.task_name,
			cache_dir = args.cache_dir if args.cache_dir else None,
		)

		config.output_attentions = True

		model = model_cls.from_pretrained(
			checkpoint,
			from_tf   = bool('.ckpt' in args.model_name_or_path),
			config    = config,
			cache_dir = args.cache_dir if args.cache_dir else None,
		)

		model.to(args.device)

		vscore, vprob = visualize(
			args         = args,
			model        = model,
			tokenizer    = tokenizer,
			prefix       = prefix,
			kmer         = kmer,
			use_features = use_features
		)

		if scores is not None :
			probs  = probs  + vprob
			scores = scores + vscore
		else :
			probs  = deepcopy(vprob)
			scores = deepcopy(vscore)

	probs = probs / float(len(models))

	numpy.save(os.path.join(args.predict_dir, 'atten.npy'), scores)
	numpy.save(os.path.join(args.predict_dir, 'probs.npy'), probs)

def bert_ensamble (args : Any, model_cls : Any, tokenizer_cls : Any, config_cls : Any, num_labels : int, logger : Optional[Any], use_features : bool = False) -> None :
	"""
	Doc
	"""

	if not (args.do_ensemble_pred and args.local_rank in [-1, 0]) :
		return

	out_label_ids = None
	eval_task     = None
	all_probs     = None
	cat_probs     = None
	prefix        = ''

	for kmer in range(3, 7) :
		output_dir = os.path.join(args.output_dir, str(kmer))

		tokenizer = tokenizer_cls.from_pretrained(
			'dna' + str(kmer),
			do_lower_case = args.do_lower_case,
			cache_dir     = args.cache_dir if args.cache_dir else None
		)

		checkpoint = output_dir

		if logger is not None :
			logger.info('Calculate attention score using the following checkpoint: %s', checkpoint)

		if checkpoint.find('checkpoint') != -1 :
			prefix = checkpoint.split('/')[-1]
		else :
			prefix = ''

		config = config_cls.from_pretrained(
			output_dir,
			num_labels      = num_labels,
			finetuning_task = args.task_name,
			cache_dir       = args.cache_dir if args.cache_dir else None
		)

		config.output_attentions = True

		model = model_cls.from_pretrained(
			args.model_name_or_path,
			from_tf   = bool('.ckpt' in args.model_name_or_path),
			config    = config,
			cache_dir = args.cache_dir if args.cache_dir else None,
		)

		model.to(args.device)

		if kmer == 3 :
			args.data_dir = os.path.join(args.data_dir, str(kmer))
		else :
			args.data_dir = args.data_dir.replace('/' + str(kmer - 1), '/' + str(kmer))

		if args.result_dir.split('/')[-1] == 'test.npy' :
			results, eval_task, _, out_label_ids, probs = evaluate(
				args         = args,
				model        = model,
				tokenizer    = tokenizer,
				prefix       = prefix,
				use_features = use_features
			)
		elif args.result_dir.split('/')[-1] == 'train.npy' :
			results, eval_task, _, out_label_ids, probs = evaluate(
				args            = args,
				model           = model,
				tokenizer       = tokenizer,
				prefix          = prefix,
				should_evaluate = False,
				use_features    = use_features
			)
		else :
			raise ValueError('File name in result_dir should be either test.npy or train.npy')

		if kmer == 3 :
			all_probs = deepcopy(probs)
			cat_probs = deepcopy(probs)
		else :
			all_probs += probs
			cat_probs = numpy.concatenate((cat_probs, probs), axis = 1)

	all_probs = all_probs / 4.0
	all_preds = numpy.argmax(all_probs, axis = 1)

	labels = numpy.array(out_label_ids)
	labels = labels.reshape(labels.shape[0], 1)
	data   = numpy.concatenate((cat_probs, labels), axis = 1)

	random.shuffle(data) # noqa

	root_path = args.result_dir.replace(args.result_dir.split('/')[-1], '')

	if not os.path.exists(root_path) :
		os.makedirs(root_path)

	numpy.save(args.result_dir, data)

	ensemble_results = compute_metrics(
		task_name = eval_task,
		preds     = all_preds,
		labels    = out_label_ids,
		probs     = all_probs[:, 1]
	)

	if logger is not None :
		logger.info('***** Ensemble results {} *****'.format(prefix))

		for key in sorted(ensemble_results.keys()) :
			logger.info('  %s = %s', key, str(ensemble_results[key]))
