from torch  import Tensor
from typing import Dict

import numpy
import os

def process_extraction_result (inputs : Dict[str, Tensor], outputs : Tensor, batch_index : int, directory : str, mode : str = 'train') -> None :
	"""
	Doc
	"""

	names = [
		'hidden_state_last', # output[0] =       tensor | last_hidden_state | batch_size, seq_len, hidden_size
		'features',          # output[1] =       tensor | pooler_output     | batch_size, hidden_size
		'hidden_state',      # output[2] = tuple tensor | hidden_states     | batch_size, seq_len, hidden_size      | config.output_hidden_states = True
		'attention'          # output[3] = tuple tensor | attentions        | batch_size, n_heads, seq_len, seq_len | config.output_attentions    = True
	]

	for index, output in enumerate(outputs) :
		if index == 1 :
			process_extracion_result_single(
				inputs      = inputs,
				outputs     = output,
				batch_index = batch_index,
				directory   = directory,
				name        = names[index],
				mode        = mode
			)

def process_extracion_result_single (inputs :  Dict[str, Tensor], outputs : Tensor, batch_index : int, directory : str, name : str, mode : str = 'train') -> None :
	"""
	Doc
	"""

	mode = mode.lower()

	if mode not in ['dev', 'train'] :
		raise ValueError()

	labels   = inputs.get('labels',   None)
	features = inputs.get('features', None)

	if   labels is not None and isinstance(  labels, Tensor) :   labels =   labels.detach().cpu().numpy()
	if features is not None and isinstance(features, Tensor) : features = features.detach().cpu().numpy()
	if  outputs is not None and isinstance( outputs, Tensor) :  outputs =  outputs.detach().cpu().numpy()

	batch_file = os.path.join(directory, '{}_{}_{}'.format(mode, name, batch_index))
	batch_data = {
		'outputs'  : outputs,
		'features' : features,
		'labels'   : labels
	}

	numpy.savez(batch_file, **batch_data)
