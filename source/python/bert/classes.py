from transformers import BertModel           # noqa F821 :: unresolved reference :: added at runtime
from transformers import BertPreTrainedModel # noqa F821 :: unresolved reference :: added at runtime
from transformers import DataProcessor       # noqa F821 :: unresolved reference :: added at runtime
from transformers import InputExample        # noqa F821 :: unresolved reference :: added at runtime

from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import MSELoss

import os

class RegressionBertFC1 (BertPreTrainedModel) :
	"""
	transformers.modeling_bert.BertModel()
	transformers.modeling_bert.BertPreTrainedModel()
	transformers.modeling_bert.BertForSequenceClassification()
	"""

	def __init__ (self, config) :
		"""
		Doc
		"""

		super().__init__(config)

		self.num_labels = config.num_labels

		self.bert    = BertModel(config)
		self.dropout = Dropout(config.hidden_dropout_prob)
		self.fc      = Linear(config.hidden_size, self.config.num_labels)

	def forward (self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None) :
		"""
		Doc
		"""

		outputs = self.bert(
			input_ids,
			attention_mask = attention_mask,
			token_type_ids = token_type_ids,
			position_ids   = position_ids,
			head_mask      = head_mask,
			inputs_embeds  = inputs_embeds
		)

		output = outputs[1]

		output = self.dropout(output)
		logits = self.fc(output)

		# Add hidden states and attention if they are here
		outputs = (logits,) + outputs[2:]

		if labels is not None :
			if self.num_labels == 1 :
				loss = MSELoss()
				loss = loss(logits.view(-1), labels.view(-1))
			else :
				loss = MSELoss()
				loss = loss(logits.view(-1), labels.view(-1))

			outputs = (loss,) + outputs

		# (loss), logits, (hidden_states), (attentions)
		return outputs

class RegressionProcessor (DataProcessor) :
	"""
	transformers.data.processors.utils.DataProcessor()
	transformers.data.processors.glue.DnaPromProcessor()
	"""

	def get_example_from_tensor_dict (self, tensor_dict) :
		"""
		Doc
		"""

		print(tensor_dict)
		raise NotImplementedError()

	def get_train_examples (self, data_dir) :
		"""
		Doc
		"""

		return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')

	def get_dev_examples (self, data_dir) :
		"""
		Doc
		"""

		return self._create_examples(self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

	def get_labels (self) : # noqa U100 :: method may be static
		"""
		Doc
		"""

		return [None]

	def _create_examples (self, lines, set_type) : # noqa U100 :: method may be static
		"""
		Doc
		"""

		examples = []

		for (i, line) in enumerate(lines) :
			if i == 0 : continue

			guid  = '%s-%s' % (set_type, i)
			text  = line[0]
			label = line[1]

			examples.append(InputExample(guid = guid, text_a = text, text_b = None, label = label))

		return examples
