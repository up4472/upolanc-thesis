from transformers import BertModel           # noqa F821 :: unresolved reference :: added at runtime
from transformers import BertPreTrainedModel # noqa F821 :: unresolved reference :: added at runtime

from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import MSELoss

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
		self.fc1     = Linear(config.hidden_size, self.config.num_labels)

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

		logits = outputs[1]

		logits = self.dropout(logits)
		logits = self.fc1(logits)

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

class RegressionBertFC3 (BertPreTrainedModel) :
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

		self.fc1 = Linear(config.hidden_size, 512)
		self.fc2 = Linear(512,                256)
		self.fc3 = Linear(256,                self.config.num_labels)

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

		logits = outputs[1]

		logits = self.dropout(logits)
		logits = self.fc1(logits)
		logits = self.dropout(logits)
		logits = self.fc2(logits)
		logits = self.dropout(logits)
		logits = self.fc3(logits)

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
