from transformers import BertModel           # noqa F821 :: unresolved reference :: added at runtime
from transformers import BertPreTrainedModel # noqa F821 :: unresolved reference :: added at runtime

from torch.nn import Dropout
from torch.nn import GRU
from torch.nn import LSTM
from torch.nn import LeakyReLU
from torch.nn import Linear
from torch.nn import MSELoss

import torch

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

		self.bert    = BertModel(config)
		self.dropout = Dropout(config.hidden_dropout_prob)

		if config.use_features :
			idim = config.hidden_size + config.num_features
			odim = config.num_labels
		else :
			idim = config.hidden_size
			odim = config.num_labels

		self.fc1 = Linear(idim, odim)

	def forward (self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, features = None) :
		"""
		Doc
		"""

		outputs = self.bert(
			input_ids      = input_ids,
			attention_mask = attention_mask,
			token_type_ids = token_type_ids,
			position_ids   = position_ids,
			head_mask      = head_mask,
			inputs_embeds  = inputs_embeds
		)

		logits = outputs[1]

		logits = self.dropout(logits)

		if features is not None :
			logits = torch.cat(
				tensors = (logits, features),
				dim     = 1
			)

		logits = self.fc1(logits)

		outputs = (logits,) + outputs[2:]

		if labels is not None :
			loss = MSELoss()
			loss = loss(logits.view(-1), labels.view(-1))

			outputs = (loss,) + outputs

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

		self.bert    = BertModel(config)
		self.dropout = Dropout(config.hidden_dropout_prob)

		self.relu    = LeakyReLU(
			negative_slope = 0.0,
			inplace        = False
		)

		if config.use_features :
			idim = config.hidden_size + config.num_features
			odim = config.num_labels
		else :
			idim = config.hidden_size
			odim = config.num_labels

		self.fc1 = Linear(idim, 512)
		self.fc2 = Linear(512,  256)
		self.fc3 = Linear(256,  odim)

	def forward (self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, features = None) :
		"""
		Doc
		"""

		outputs = self.bert(
			input_ids      = input_ids,
			attention_mask = attention_mask,
			token_type_ids = token_type_ids,
			position_ids   = position_ids,
			head_mask      = head_mask,
			inputs_embeds  = inputs_embeds
		)

		logits = outputs[1]
		logits = self.dropout(logits)

		if features is not None :
			logits = torch.cat(
				tensors = (logits, features),
				dim     = 1
			)

		logits = self.fc1(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.fc2(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.fc3(logits)

		outputs = (logits,) + outputs[2:]

		if labels is not None :
			loss = MSELoss()
			loss = loss(logits.view(-1), labels.view(-1))

			outputs = (loss,) + outputs

		return outputs

class CatRegressionBertFC3 (BertPreTrainedModel) :
	"""
	transformers.modeling_bert.BertModel()
	transformers.modeling_bert.BertPreTrainedModel()
	transformers.modeling_bert.BertForLongSequenceClassificationCat()
	BertForLongSequenceClassification
	"""

	def __init__ (self, config) :
		"""
		Doc
		"""

		super().__init__(config)

		self.num_labels = config.num_labels
		self.split      = config.split

		self.bert    = BertModel(config)
		self.dropout = Dropout(config.hidden_dropout_prob)

		self.relu    = LeakyReLU(
			negative_slope = 0.0,
			inplace        = False
		)

		xdim = config.hidden_size * self.split
		ydim = config.hidden_size

		if config.use_features :
			idim = config.hidden_size + config.num_features
			odim = config.num_labels
		else :
			idim = config.hidden_size
			odim = config.num_labels

		self.fc1 = Linear(xdim, ydim)
		self.fc2 = Linear(idim, 512)
		self.fc3 = Linear(512,  256)
		self.fc4 = Linear(256,  odim)

		self.init_weights()

	def forward (self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, overlap = 100, max_length_per_seq = 500, features = None) : # noqa : unused parameters
		"""
		Doc
		"""

		batch_size     = input_ids.shape[0]
		input_ids      = input_ids.view(self.split * batch_size, 512)
		attention_mask = attention_mask.view(self.split * batch_size, 512)

		outputs = self.bert(
			input_ids      = input_ids,
			attention_mask = attention_mask,
			token_type_ids = None,
			position_ids   = position_ids,
			head_mask      = head_mask,
			inputs_embeds  = inputs_embeds
		)

		logits = outputs[1].view(batch_size, -1)
		logits = self.dropout(logits)

		logits = self.fc1(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)

		logits = self.fc2(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.fc3(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.fc4(logits)

		outputs = (logits,) + outputs[2:]

		if labels is not None :
			loss = MSELoss()
			loss = loss(logits.view(-1), labels.view(-1))

			outputs = (loss,) + outputs

		return outputs

class RnnRegressionBertFC3 (BertPreTrainedModel) :
	"""
	transformers.modeling_bert.BertModel()
	transformers.modeling_bert.BertPreTrainedModel()
	transformers.modeling_bert.BertForLongSequenceClassification()
	"""

	def __init__ (self, config) :
		"""
		Doc
		"""

		super().__init__(config)

		self.num_labels  = config.num_labels
		self.split       = config.split
		self.hidden_size = config.hidden_size
		self.rnn_type    = config.rnn
		self.rnn_layers  = config.num_rnn_layer
		self.rnn_dropout = config.rnn_dropout
		self.rnn_hidden  = config.rnn_hidden

		self.bert    = BertModel(config)
		self.dropout = Dropout(config.hidden_dropout_prob)

		if   self.rnn_type == 'lstm' : self.rnn = LSTM
		elif self.rnn_type == 'gru'  : self.rnn = GRU
		else : raise ValueError()

		self.rnn = self.rnn(
			input_size    = self.hidden_size,
			hidden_size   = self.hidden_size,
			bidirectional = True,
			num_layers    = self.rnn_layers,
			batch_first   = True,
			dropout       = self.rnn_dropout
		)

		self.relu    = LeakyReLU(
			negative_slope = 0.0,
			inplace        = False
		)

		if config.use_features :
			idim = config.hidden_size + config.num_features
			odim = config.num_labels
		else :
			idim = config.hidden_size
			odim = config.num_labels

		self.fc1 = Linear(idim, 512)
		self.fc2 = Linear(512,  256)
		self.fc3 = Linear(256,  odim)

		self.init_weights()

	def forward (self, input_ids = None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask = None, inputs_embeds = None, labels = None, overlap = 100, max_length_per_seq = 500, features = None) : # noqa : unused parameters
		"""
		Doc
		"""

		batch_size     = input_ids.shape[0]
		input_ids      = input_ids.view(self.split * batch_size, 512)
		attention_mask = attention_mask.view(self.split * batch_size, 512)

		outputs = self.bert(
			input_ids      = input_ids,
			attention_mask = attention_mask,
			token_type_ids = None,
			position_ids   = position_ids,
			head_mask      = head_mask,
			inputs_embeds  = inputs_embeds
		)

		logits = outputs[1].view(batch_size, self.split, self.hidden_size)

		if   self.rnn_type == 'lstm' : _, (ht, _)   = self.rnn(logits)
		elif self.rnn_type == 'gru'  : _, ht        = self.rnn(logits)
		else : raise ValueError()

		logits = self.dropout(ht.squeeze(0).sum(dim = 0))

		if features is not None :
			logits = torch.cat(
				tensors = (logits, features),
				dim     = 1
			)

		logits = self.fc1(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.fc2(logits)
		logits = self.relu(logits)
		logits = self.dropout(logits)
		logits = self.fc3(logits)

		outputs = (logits,) + outputs[2:]

		if labels is not None :
			loss = MSELoss()
			loss = loss(logits.view(-1), labels.view(-1))

			outputs = (loss,) + outputs

		return outputs
