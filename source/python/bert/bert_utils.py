from transformers import AdamW           # noqa F821 :: unresolved reference :: added at runtime
from transformers import InputExample    # noqa F821 :: unresolved reference :: added at runtime
from transformers import InputFeatures   # noqa F821 :: unresolved reference :: added at runtime
from transformers import is_tf_available # noqa F821 :: unresolved reference :: added at runtime

from source.python.bert.bert_constants import MODES
from source.python.bert.bert_constants import PROCESSORS
from source.python.bert.bert_input     import BertFeatures

if is_tf_available() : import tensorflow # noqa

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

def bert_convert_examples_to_features (examples, tokenizer, max_length = 512, task = None, label_list = None, output_mode = None, pad_on_left = False, pad_token = 0, pad_token_segment_id = 0, mask_padding_with_zero = True) :
	"""
	Doc
	"""

	is_tf_dataset = False
	processor     = None
	features      = []

	if is_tf_available() and isinstance(examples, tensorflow.data.Dataset):
		is_tf_dataset = True

	if task is not None :
		processor = PROCESSORS[task]()

		if label_list is None :
			label_list = processor.get_labels()
		if output_mode is None :
			output_mode = MODES[task]

	label_map = {
		label : i
		for i, label in enumerate(label_list)
	}

	for (index, example) in enumerate(examples) :
		if is_tf_dataset and processor is not None :
			example = processor.get_example_from_tensor_dict(example)
			example = processor.tfds_map(example)

		inputs = tokenizer.encode_plus(
			example.text_a,
			example.text_b,
			add_special_tokens = True,
			max_length         = max_length
		)

		input_ids      = inputs['input_ids']
		token_type_ids = inputs['token_type_ids']

		#
		# The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
		#

		attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
		padding_length = max_length - len(input_ids)

		if pad_on_left :
			input_ids      = ([pad_token] * padding_length) + input_ids
			attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
			token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
		else :
			input_ids      = input_ids + ([pad_token] * padding_length)
			attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
			token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

		assert len(input_ids) == max_length,      'Error with input length {} vs {}'.format(len(input_ids), max_length)
		assert len(attention_mask) == max_length, 'Error with input length {} vs {}'.format(len(attention_mask), max_length)
		assert len(token_type_ids) == max_length, 'Error with input length {} vs {}'.format(len(token_type_ids), max_length)

		if   output_mode == 'classification' : label = label_map[example.label]
		elif output_mode == 'regression'     : label = float(example.label)
		else                                 : raise KeyError(output_mode)

		features.append(BertFeatures(
			input_ids      = input_ids,
			attention_mask = attention_mask,
			token_type_ids = token_type_ids,
			label          = label,
			feature        = example.feature
		))

	if is_tf_available () and is_tf_dataset :
		def gen () :
			for ex in features :
				yield (
					{
						'input_ids'      : ex.input_ids,
						'attention_mask' : ex.attention_mask,
						'token_type_ids' : ex.token_type_ids,
					},
					ex.label,
					ex.feature
				)

		return tensorflow.data.Dataset.from_generator(
			gen,
			(
				{
					'input_ids'      : tensorflow.int32,
					'attention_mask' : tensorflow.int32,
					'token_type_ids' : tensorflow.int32
				},
				tensorflow.int64,
				tensorflow.int64
			),
			(
				{
					'input_ids'      : tensorflow.TensorShape([None]),
					'attention_mask' : tensorflow.TensorShape([None]),
					'token_type_ids' : tensorflow.TensorShape([None]),
				},
				tensorflow.TensorShape([]),
				tensorflow.TensorShape([])
			)
		)

	return features
