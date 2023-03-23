from transformers import DataProcessor # noqa F821 :: unresolved reference :: added at runtime
from transformers import InputExample  # noqa F821 :: unresolved reference :: added at runtime

import os

from source.python.bert.bert_input import BertInput

class RegressionProcessor (DataProcessor) :
	"""
	transformers.data.processors.utils.DataProcessor()
	transformers.data.processors.glue.DnaPromProcessor()
	"""

	def get_example_from_tensor_dict (self, tensor_dict) :
		"""
		Doc
		"""

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

	@staticmethod
	def to_float_array_or_float (value) :
		"""
		Doc
		"""

		if value is not None :
			value = value.replace('[', '')
			value = value.replace(']', '')
			value = value.split()

			value = [float(x) for x in value]

			if len(value) == 1 :
				value = value[0]

		return value

	def _create_examples (self, lines, set_type) : # noqa U100 :: method may be static
		"""
		Doc
		"""

		examples = []

		for (i, line) in enumerate(lines) :
			if i == 0 : continue

			guid  = '%s-%s' % (set_type, i)
			text    = line[0]
			label   = line[1]

			if len(line) >= 3 : feature = line[2]
			else              : feature = None

			label   = RegressionProcessor.to_float_array_or_float(value = label)
			feature = RegressionProcessor.to_float_array_or_float(value = feature)

			examples.append(BertInput(
				guid    = guid,
				text_a  = text,
				text_b  = None,
				label   = label,
				feature = feature
			))

		return examples
