from torch.nn         import Module
from torch.utils.data import TensorDataset
from typing           import Any

from tqdm import tqdm
from tqdm import trange

import logging
import numpy
import os
import random
import torch
import torch.distributed

from source.python.bert.bert_cache      import load_and_cache_examples
from source.python.bert.bert_extraction import process_extraction_result
from source.python.bert.bert_helper     import log_result
from source.python.bert.bert_helper     import prepare_bert_inputs
from source.python.bert.bert_helper     import process_bert_clip_grad
from source.python.bert.bert_helper     import process_bert_loss
from source.python.bert.bert_helper     import process_results
from source.python.bert.bert_helper     import save_model_checkpoint
from source.python.bert.bert_utils      import freeze_bert
from source.python.bert.bert_utils      import get_dataloader
from source.python.bert.bert_utils      import get_optimizer
from source.python.bert.bert_utils      import get_scheduler
from source.python.bert.bert_utils      import prepare_eval_model
from source.python.bert.bert_utils      import prepare_train_model

logger = logging.getLogger(__name__)

def set_seed (args : Any) -> None :
	"""
	Doc
	"""

	random.seed(args.seed)
	numpy.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if args.n_gpu > 0 :
		torch.cuda.manual_seed_all(args.seed)

def train (args : Any, train_dataset : TensorDataset, model : Module, tokenizer : Any, use_features : bool = False) :
	"""
	Doc
	"""

	train_dataloader = get_dataloader(
		dataset    = train_dataset,
		mode       = 'train',
		local_rank = args.local_rank,
		batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	)

	optimizer = get_optimizer(
		model = model,
		args  = args
	)

	scheduler = get_scheduler(
		optimizer  = optimizer,
		dataloader = train_dataloader,
		args       = args
	)

	can_load_optimizer = os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt'))
	can_load_scheduler = os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt'))

	if can_load_optimizer and can_load_scheduler :
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

	model = prepare_train_model(
		model     = model,
		optimizer = optimizer,
		args      = args
	)

	#
	# Freeze certain layers
	#

	freeze_bert(
		model  = model,
		layers = args.freeze_layers
	)

	#
	# Check for checkpoints
	#

	global_step = 0
	epochs_div  = 0
	epochs_mod  = 0

	if os.path.exists(args.model_name_or_path) :
		if '-' in args.model_name_or_path :
			global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])

		x = len(train_dataloader) // args.gradient_accumulation_steps

		epochs_div = global_step // x
		epochs_mod = global_step %  x

	#
	# Start training
	#

	logger.info('***** Running training *****')

	logging_reports = []
	running_loss    = 0.0
	logging_loss    = 0.0

	model.zero_grad()
	set_seed(args)

	train_iterator = trange(epochs_div, int(args.num_train_epochs), desc = 'Epoch', disable = True)

	for _ in train_iterator :
		epoch_iterator = tqdm(train_dataloader, desc = 'Iteration', disable = True)

		for step, batch in enumerate(epoch_iterator) :
			if epochs_mod > 0 :
				epochs_mod = epochs_mod - 1
				continue

			model.train()

			inputs = prepare_bert_inputs(
				model_type = args.model_type,
				batch      = tuple(t.to(args.device) for t in batch)
			)

			outputs = model(**inputs)

			loss = outputs[0]
			loss = process_bert_loss(
				loss      = loss,
				optimizer = optimizer,
				args      = args
			)

			running_loss = running_loss + loss.item()

			if (step + 1) % args.gradient_accumulation_steps == 0 :
				process_bert_clip_grad(
					model     = model,
					optimizer = optimizer,
					args      = args
				)

				optimizer.step()
				scheduler.step()
				model.zero_grad()

				global_step = global_step + 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 :
					logs = {}

					if args.local_rank == -1 and args.evaluate_during_training :
						results = evaluate(
							args         = args,
							model        = model,
							tokenizer    = tokenizer,
							use_features = use_features
						)

						for key, value in results.items() :
							eval_key = 'eval_{}'.format(key)
							logs[eval_key] = value

					loss_scalar = (running_loss - logging_loss) / args.logging_steps
					learning_rate_scalar = scheduler.get_lr()[0]

					logs['learning_rate'] = learning_rate_scalar
					logs['loss'] = loss_scalar

					logging_loss   = running_loss
					logging_report = {**logs, **{'step' : global_step}}
					logging_reports.append(logging_report)

				save_model_checkpoint(
					model       = model,
					tokenizer   = tokenizer,
					optimizer   = optimizer,
					scheduler   = scheduler,
					args        = args,
					global_step = global_step
				)

			if 0 < args.max_steps < global_step :
				epoch_iterator.close()
				break

		if 0 < args.max_steps < global_step :
			train_iterator.close()
			break

	return global_step, running_loss / global_step, logging_reports

def evaluate (args : Any, model : Module, tokenizer : Any, prefix : str = '', should_evaluate : bool = True, use_features : bool = False) :
	"""
	Doc
	"""

	#
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	#

	if args.task_name == 'mnli' :
		task_names  = ('mnli', 'mnli-mm')
		output_dirs = (args.output_dir, args.output_dir + '-MM')
	else :
		task_names  = (args.task_name,)
		output_dirs = (args.output_dir,)

	task_name = None
	ypred     = None
	yprob     = None
	ytrue     = None
	results   = {}

	for task_name, output_dir in zip(task_names, output_dirs) :
		eval_dataset = load_and_cache_examples(
			args            = args,
			task            = task_name,
			tokenizer       = tokenizer,
			should_evaluate = should_evaluate,
			use_features    = use_features
		)

		if not os.path.exists(output_dir) and args.local_rank in [-1, 0] :
			os.makedirs(output_dir)

		eval_dataloader = get_dataloader(
			dataset    = eval_dataset,
			mode       = 'eval',
			local_rank = args.local_rank,
			batch_size = args.eval_batch_size
		)

		model = prepare_eval_model(
			model = model,
			args  = args
		)

		#
		# Start evaluation
		#

		logger.info('***** Running evaluation {} *****'.format(prefix))

		running_loss = 0.0
		ypred        = None
		ytrue        = None

		for batch in tqdm(eval_dataloader, desc = 'Evaluating', disable = True) :
			model.eval()

			with torch.no_grad() :
				inputs = prepare_bert_inputs(
					model_type = args.model_type,
					batch      = tuple(t.to(args.device) for t in batch)
				)

				outputs = model(**inputs)
				loss   = outputs[0]
				logits = outputs[1]

				running_loss =  running_loss + loss.mean().item()

			_ypred = logits.detach().cpu().numpy()
			_ytrue = inputs['labels'].detach().cpu().numpy()

			if ypred is None :
				ypred = _ypred
				ytrue = _ytrue
			else :
				ypred = numpy.append(ypred, _ypred, axis = 0)
				ytrue = numpy.append(ytrue, _ytrue, axis = 0)

		result = process_results(
			task_name = task_name,
			preds     = ypred,
			labels    = ytrue,
			args      = args
		)

		results.update(result)

		log_result(
			output_dir = output_dir,
			prefix     = prefix,
			args       = args,
			result     = result,
			mode       = 'eval',
			logger     = logger
		)

	if args.do_ensemble_pred :
		return results, task_name, ypred, ytrue, yprob

	return results

def predict (args : Any, model : Module, tokenizer : Any, prefix : str = '', use_features : bool = False) :
	"""
	Doc
	"""

	#
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	#

	task_names  = (args.task_name,)
	output_dirs = (args.predict_dir,)

	if not os.path.exists(args.predict_dir) :
		os.makedirs(args.predict_dir)

	for task_name, output_dir in zip(task_names, output_dirs) :
		pred_dataset = load_and_cache_examples(
			args            = args,
			task            = task_name,
			tokenizer       = tokenizer,
			should_evaluate = True,
			use_features    = use_features
		)

		if not os.path.exists(output_dir) and args.local_rank in [-1, 0] :
			os.makedirs(output_dir)

		pred_dataloader = get_dataloader(
			dataset    = pred_dataset,
			mode       = 'predict',
			local_rank = args.local_rank,
			batch_size = args.pred_batch_size
		)

		model = prepare_eval_model(
			model = model,
			args  = args
		)

		#
		# Start prediction
		#

		logger.info('***** Running prediction {} *****'.format(prefix))

		ypred = None
		ytrue = None

		for batch in tqdm(pred_dataloader, desc = 'Predicting', disable = True) :
			model.eval()
			batch = tuple(t.to(args.device) for t in batch)

			with torch.no_grad() :
				inputs = prepare_bert_inputs(
					model_type = args.model_type,
					batch      = batch
				)

				outputs = model(**inputs)
				logits  = outputs[1]

			_ypred = logits.detach().cpu().numpy()
			_ytrue = inputs['labels'].detach().cpu().numpy()

			if ypred is None :
				ypred = _ypred
				ytrue = _ytrue
			else :
				ypred = numpy.append(ypred, _ypred, axis = 0)
				ytrue = numpy.append(ytrue, _ytrue, axis = 0)

		result = process_results(
			task_name = task_name,
			preds     = ypred,
			labels    = ytrue,
			args      = args
		)

		log_result(
			output_dir = args.predict_dir,
			prefix     = prefix,
			args       = args,
			result     = result,
			mode       = 'pred',
			logger     = logger
		)

		return result

def extract (args : Any, model : Module, dataset : TensorDataset, mode : str = 'train') -> None :
	"""
	Doc
	"""

	dataloader = get_dataloader(
		dataset    = dataset,
		mode       = mode,
		local_rank = args.local_rank,
		batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	)

	optimizer = get_optimizer(
		model = model,
		args  = args
	)

	scheduler = get_scheduler(
		optimizer  = optimizer,
		dataloader = dataloader,
		args       = args
	)

	can_load_optimizer = os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt'))
	can_load_scheduler = os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt'))

	if can_load_optimizer and can_load_scheduler :
		optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
		scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

	model = prepare_train_model(
		model     = model,
		optimizer = optimizer,
		args      = args
	)

	#
	# Start extraction
	#

	logger.info('***** Running extraction {} *****'.format(mode))

	for index, batch in enumerate(tqdm(dataloader, desc = 'Extracting', disable = True)) :
		model.eval()

		with torch.no_grad() :
			inputs = prepare_bert_inputs(
				model_type = args.model_type,
				batch      = tuple(t.to(args.device) for t in batch)
			)

			outputs = model(**inputs)

			process_extraction_result(
				inputs      = inputs,
				outputs     = outputs,
				batch_index = index,
				directory   = args.output_dir,
				mode        = mode
			)

def visualize (args : Any, model : Module, tokenizer : Any, kmer : int, prefix : str = '', use_features : bool = False) :
	"""
	Doc
	"""

	#
	# Loop to handle MNLI double evaluation (matched, mis-matched)
	#

	task_names  = (args.task_name,)
	output_dirs = (args.predict_dir,)

	if not os.path.exists(args.predict_dir) :
		os.makedirs(args.predict_dir)

	score = None
	yprob = None

	for task_name, output_dir in zip(task_names, output_dirs) :
		pred_dataset = load_and_cache_examples(
			args            = args,
			task            = task_name,
			tokenizer       = tokenizer,
			should_evaluate = False if args.visualize_train else True,
			use_features    = use_features
		)

		if not os.path.exists(output_dir) and args.local_rank in [-1, 0] :
			os.makedirs(output_dir)

		pred_dataloader = get_dataloader(
			dataset    = pred_dataset,
			mode       = 'predict',
			local_rank = args.local_rank,
			batch_size = args.pred_batch_size
		)

		model = prepare_eval_model(
			model = model,
			args  = args
		)

		#
		# Start visualization
		#

		logger.info('***** Running visualization {} *****'.format(prefix))

		if args.task_name != 'dnasplice' :
			ypred = numpy.zeros([len(pred_dataset), 2])
			yattn = numpy.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])
		else :
			ypred = numpy.zeros([len(pred_dataset), 3])
			yattn = numpy.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])

		for index, batch in enumerate(tqdm(pred_dataloader, desc = 'Visualization')) :
			model.eval()

			with torch.no_grad() :
				inputs = prepare_bert_inputs(
					model_type = args.model_type,
					batch      = tuple(t.to(args.device) for t in batch)
				)

				outputs   = model(**inputs)
				attention = outputs[-1][-1]
				logits    = outputs[1]

				_ypred = logits.detach().cpu().numpy()
				_yattn = attention.cpu().numpy()

				xs = index * args.pred_batch_size
				xe = index * args.pred_batch_size + len(batch[0])

				ypred[xs:xe, :]       = _ypred
				yattn[xs:xe, :, :, :] = _yattn

		#
		# Alaways applies Softmax no mather the task (so not good for regression)
		#

		if args.task_name != 'dnasplice' :
			yprob = torch.tensor(ypred, dtype = torch.float32)
			yprob = torch.softmax(yprob, dim = 1)[:, 1].numpy()
		else :
			yprob = torch.tensor(ypred, dtype = torch.float32)
			yprob = torch.softmax(yprob, dim = 1).numpy()

		score = numpy.zeros([yattn.shape[0], yattn.shape[-1]])

		for index, attention_score in enumerate(yattn) :
			attn_score = []

			for i in range(1, attention_score.shape[-1] - kmer + 2) :
				attn_score.append(float(attention_score[:, 0, i].sum()))

			for i in range(len(attn_score) - 1) :
				if attn_score[i + 1] == 0 :
					attn_score[i] = 0
					break

			counts = numpy.zeros([len(attn_score) + kmer - 1])
			scores = numpy.zeros([len(attn_score) + kmer - 1])

			for i, score in enumerate(attn_score) :
				for j in range(kmer) :
					counts[i + j] = counts[i + j] + 1.0
					scores[i + j] = scores[i + j] + score

			scores = scores / counts
			scores = scores / numpy.linalg.norm(scores)

			score[index] = scores

	return score, yprob
