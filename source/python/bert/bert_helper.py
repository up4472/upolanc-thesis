from torch       import Tensor
from torch.nn    import Module
from torch.optim import Optimizer
from typing      import Any
from typing      import Dict
from typing      import Optional

from torch.nn.utils import clip_grad_norm_

import numpy
import os
import torch

from source.python.bert.bert_checkpoint import rotate_checkpoints
from source.python.bert.bert_constants  import TOKENS
from source.python.bert.bert_metrics    import compute_metrics

def prepare_bert_inputs (model_type : str, batch : Any) -> Dict[str, Any] :
	"""
	Doc
	"""

	inputs = {
		'input_ids'      : batch[0],
		'attention_mask' : batch[1],
		'labels'         : batch[3]
	}

	if model_type != 'distilbert' :
		if model_type in TOKENS :
			inputs['token_type_ids'] = batch[2]
		else :
			inputs['token_type_ids'] = None

	if len(batch) >= 5 :
		inputs['features'] = batch[4]

	return inputs

def process_bert_loss (loss : Any, optimizer : Optimizer, args : Any) -> Any :
	"""
	Doc
	"""

	if args.n_gpu > 1 :
		loss = loss.mean()

	if args.gradient_accumulation_steps > 1 :
		loss = loss / args.gradient_accumulation_steps

	if args.fp16 :
		try                : from apex import amp # noqa
		except ImportError : raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')

		with amp.scale_loss(loss, optimizer) as scaled_loss :
			scaled_loss.backward()
	else :
		loss.backward()

	return loss

def process_bert_clip_grad (model : Module, optimizer : Optimizer, args : Any) -> Tensor :
	"""
	Doc
	"""

	if args.fp16 :
		try :                from apex import amp  # noqa
		except ImportError : raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')

		return clip_grad_norm_(
			parameters = amp.master_params(optimizer),
			max_norm   = args.max_grad_norm
		)
	else :
		return clip_grad_norm_(
			parameters = model.parameters(),
			max_norm   = args.max_grad_norm
		)

def process_results (task_name : str, preds : Any, labels : Any, args : Any) -> Dict[str, Any] :
	"""
	Doc
	"""

	probs = None

	if args.output_mode == 'classification' :
		if args.task_name[:3] == 'dna' and args.task_name != 'dnasplice' :
			if args.do_ensemble_pred :
				probs = torch.tensor(preds, dtype = torch.float32)
				probs = torch.softmax(probs, dim = 1).numpy()
			else :
				probs = torch.tensor(preds, dtype = torch.float32)
				probs = torch.softmax(probs, dim = 1)[:, 1].numpy()

		elif args.task_name == 'dnasplice' :
			probs = torch.tensor(preds, dtype = torch.float32)
			probs = torch.softmax(probs, dim = 1).numpy()

		preds = numpy.argmax(preds, axis = 1)

	elif args.output_mode == 'regression' :
		preds = numpy.squeeze(preds)

	if args.do_ensemble_pred :
		probs = probs[:, 1]

	return compute_metrics(
		task_name = task_name,
		preds     = preds,
		labels    = labels,
		probs     = probs
	)

def log_result (output_dir : Any, result : Any, args : Any, prefix : str, mode : str, logger : Optional[Any]) -> None :
	"""
	Doc
	"""

	if args.task_name == 'dna690' and mode != 'pred' :
		output_dir = args.result_dir

		if not os.path.exists(args.result_dir) :
			os.makedirs(args.result_dir)

	output_file = os.path.join(output_dir, prefix, mode + '_results.txt')

	with open(output_file, mode = 'a') as writer :
		if args.task_name[:3] == 'dna' :
			eval_result = args.data_dir.split('/')[-1] + ' '
		else :
			eval_result = ''

		if logger is not None :
			logger.info('***** {} results {} *****'.format(mode, prefix))

		for key in sorted(result.keys()) :
			value = str(result[key])

			if logger is not None :
				logger.info('  %s = %s', key, value)

			eval_result = eval_result + value[:5] + ' '

		writer.write(eval_result + '\n')

def save_model_checkpoint (model : Module, tokenizer : Any, optimizer : Optimizer, scheduler : Any, args : Any, global_step : int) -> None :
	"""
	Doc
	"""

	if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 :
		output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))

		if not os.path.exists(output_dir) :
			os.makedirs(output_dir)

		if hasattr(model, 'module') :
			model_to_save = model.module
		else :
			model_to_save = model

		model_to_save.save_pretrained(output_dir)
		tokenizer.save_pretrained(output_dir)

		rotate_checkpoints(args, 'checkpoint')

		if args.task_name != 'dna690' :
			torch.save(args,                   os.path.join(output_dir, 'training_args.bin'))
			torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
			torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
