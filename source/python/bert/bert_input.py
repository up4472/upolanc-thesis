from transformers import InputExample    # noqa F821 :: unresolved reference :: added at runtime
from transformers import InputFeatures   # noqa F821 :: unresolved reference :: added at runtime

import copy
import json

class BertInput (InputExample) :

	def __init__ (self, guid, text_a, text_b = None, label = None, feature = None) :
		"""
		Doc
		"""

		super().__init__(guid, text_a, text_b, label)

		self.guid    = guid
		self.text_a  = text_a
		self.text_b  = text_b
		self.label   = label
		self.feature = feature

	def __repr__ (self) :
		"""
		Doc
		"""

		return str(self.to_json_string())

	def to_dict (self) :
		"""
		Doc
		"""

		return copy.deepcopy(self.__dict__)

	def to_json_string (self) :
		"""
		Doc
		"""

		return json.dumps(self.to_dict(), indent = 2, sort_keys = True) + '\n'

class BertFeatures (InputFeatures) :

	def __init__ (self, input_ids, attention_mask = None, token_type_ids = None, label = None, feature = None) :
		"""
		Doc
		"""

		super().__init__(input_ids, attention_mask, token_type_ids, label)

		self.input_ids      = input_ids
		self.attention_mask = attention_mask
		self.token_type_ids = token_type_ids
		self.label          = label
		self.feature        = feature

	def __repr__ (self) :
		"""
		Doc
		"""

		return str(self.to_json_string())

	def to_dict (self) :
		"""
		Doc
		"""

		return copy.deepcopy(self.__dict__)

	def to_json_string (self) :
		"""
		Doc
		"""

		return json.dumps(self.to_dict(), indent = 2, sort_keys = True) + '\n'
