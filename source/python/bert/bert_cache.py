from multiprocessing  import Pool
from torch.utils.data import TensorDataset
from typing           import Any

import torch.distributed
import os

from source.python.bert.bert_constants import PROCESSORS
from source.python.bert.bert_constants import MODES
from source.python.bert.bert_input     import BertFeatures
from source.python.bert.bert_utils     import bert_convert_examples_to_features

def load_and_cache_examples (args : Any, task : str, tokenizer : Any, should_evaluate : bool = False, use_features : bool = False) :
	"""
	Doc
	"""

	#
	# Only the first process in distributed training process the dataset, and the others will use the cache
	#

	if args.local_rank not in [-1, 0] and not should_evaluate :
		torch.distributed.barrier()

	processor   = PROCESSORS[task]()
	output_mode = MODES[task]

	#
	# Load data features from cache or dataset file
	#

	t0 = 'dev' if should_evaluate else 'train'
	t1 = list(filter(None, args.model_name_or_path.split('/'))).pop()
	t2 = str(args.max_seq_length)
	t3 = str(task)

	if args.do_predict :
		cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(t0, t2, t3))
	else :
		cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(t0, t1, t2, t3))

	if os.path.exists(cached_features_file) and not args.overwrite_cache :
		features = torch.load(cached_features_file)
	else :
		label_list = processor.get_labels()

		if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta', 'xlmroberta'] :
			label_list[1], label_list[2] = label_list[2], label_list[1]

		examples = (
			processor.get_dev_examples(args.data_dir)
			if should_evaluate
			else processor.get_train_examples(args.data_dir)
		)

		#
		# Converting examples to features
		#

		max_length  = args.max_seq_length
		pad_on_left = bool(args.model_type in ['xlnet'])
		pad_token   = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

		if args.model_type in ['xlnet'] :
			pad_token_segment_id = 4
		else :
			pad_token_segment_id = 0

		if args.n_process == 1 :
			features = bert_convert_examples_to_features(
				examples    = examples,
				tokenizer   = tokenizer,
				label_list  = label_list,
				max_length  = max_length,
				task        = None,
				output_mode = output_mode,
				pad_on_left = pad_on_left,
				pad_token   = pad_token,
				pad_token_segment_id = pad_token_segment_id,
			)
		else :
			if should_evaluate :
				nproc = max(int(args.n_process / 4), 1)
			else :
				nproc = int(args.n_process)

			pool      = Pool(nproc)
			indexes   = [0]
			len_slice = int(len(examples) / nproc)

			for i in range(1, nproc + 1) :
				if i != nproc:
					indexes.append(len_slice * i)
				else :
					indexes.append(len(examples))

			results  = list()
			features = list()

			for i in range(nproc) :
				results.append(pool.apply_async(
					func = bert_convert_examples_to_features,
					args = (
						examples[indexes[i]:indexes[i + 1]],
						tokenizer, max_length, None, label_list, output_mode,
						pad_on_left, pad_token, pad_token_segment_id, True
					)
				))

			pool.close()
			pool.join()

			for result in results :
				features.extend(result.get())

		if args.local_rank in [-1, 0] :
			torch.save(features, cached_features_file)

	#
	# Only the first process in distributed training process the dataset, and the others will use the cache
	#

	if args.local_rank == 0 and not should_evaluate :
		torch.distributed.barrier()

	#
	# Convert to tensors and build dataset
	#

	all_input_ids      = torch.tensor([f.input_ids      for f in features], dtype = torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype = torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype = torch.long)
	all_labels         = None
	all_feature        = None

	if output_mode == 'classification' :
		all_labels = torch.tensor([f.label for f in features], dtype = torch.long)
	elif output_mode == 'regression' :
		all_labels = torch.tensor([f.label for f in features], dtype = torch.float)

	if use_features :
		if isinstance(features[0], BertFeatures) :
			all_feature = torch.tensor([f.feature for f in features], dtype = torch.float)
		else :
			all_feature = None

	if all_feature is not None :
		return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_feature)

	return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
