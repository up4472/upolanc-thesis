from transformers import AdamW # noqa F821 :: unresolved reference :: added at runtime

from torch.nn          import DataParallel
from torch.nn          import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim       import Optimizer
from torch.utils.data  import DataLoader
from torch.utils.data  import DistributedSampler
from torch.utils.data  import RandomSampler
from torch.utils.data  import SequentialSampler
from torch.utils.data  import TensorDataset
from typing            import Any
from typing            import Optional

from source.python.bert.dnabert.src.transformers import get_linear_schedule_with_warmup

def get_bert_module (model : Module) -> Optional[Module] :
	"""
	Doc
	"""

	name = type(model).__name__

	if   'Bert'    in name : return model.bert
	elif 'Roberta' in name : return model.roberta

	return None

def freeze_module (module : Module) -> None :
	"""
	Doc
	"""

	for param in module.parameters() :
		param.requires_grad = False

def freeze_bert (model : Module) -> None :
	"""
	Doc
	"""

	freeze_module(module = get_bert_module(model = model))

def get_sampler (dataset : TensorDataset, mode : str, local_rank : int) :
	"""
	Doc
	"""

	if mode == 'train' and local_rank == -1 : return RandomSampler(dataset)
	if mode == 'train' and local_rank != -1 : return DistributedSampler(dataset)

	return SequentialSampler(dataset)

def get_dataloader (dataset : TensorDataset, mode : str, local_rank : int, batch_size : int) -> DataLoader :
	"""
	Doc
	"""

	sampler = get_sampler(dataset = dataset, mode = mode, local_rank = local_rank)

	return DataLoader(
		dataset    = dataset,
		sampler    = sampler,
		batch_size = batch_size
	)

def get_optimizer (model : Module, args : Any) -> Optimizer :
	"""
	Doc
	"""

	optimizer_grouped_parameters = [
		{
			'params' : [
				p
				for n, p in model.named_parameters()
				if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])
			],
			'weight_decay' : args.weight_decay,
		},
		{
			'params' : [
				p
				for n, p in model.named_parameters()
				if any(nd in n for nd in ['bias', 'LayerNorm.weight'])
			],
			'weight_decay' : 0.0
		},
	]

	return AdamW(
		params = optimizer_grouped_parameters,
		lr     = args.learning_rate,
		eps    = args.adam_epsilon,
		betas  = (args.beta1, args.beta2)
	)

def get_scheduler (optimizer : Optimizer, dataloader : DataLoader, args : Any) -> Any :
	"""
	Doc
	"""

	steps_max   = args.max_steps
	steps_train = steps_max
	steps_warm  = args.warmup_steps

	x = len(dataloader) // args.gradient_accumulation_steps

	if steps_max > 0 :
		args.num_train_epochs = args.max_steps // x + 1
	else :
		steps_train = x * args.num_train_epochs

	if args.warmup_percent != 0 :
		steps_warm = int(args.warmup_percent * steps_train)

	return get_linear_schedule_with_warmup(
		optimizer          = optimizer,
		num_warmup_steps   = steps_warm,
		num_training_steps = steps_train
	)

def prepare_train_model (model : Module, optimizer : Optimizer, args : Any) -> Module :
	"""
	Doc
	"""

	if args.fp16 :
		try                : from apex import amp # noqa
		except ImportError : raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')

		model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)

	if args.n_gpu > 1 and not isinstance(model, DataParallel) :
		model = DataParallel(model)

	if args.local_rank != -1 :
		model = DistributedDataParallel(
			module        = model,
			device_ids    = [args.local_rank],
			output_device = args.local_rank,
			find_unused_parameters = True,
		)

	return model

def prepare_eval_model (model : Module, args : Any) -> Module :
	"""
	Doc
	"""

	if args.n_gpu > 1 and not isinstance(model, DataParallel) :
		model = DataParallel(model)

	return model
