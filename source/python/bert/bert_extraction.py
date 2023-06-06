import os

def process_extraction_result (inputs, outputs, directory : str, mode : str = 'append', should_evalute : bool = False) -> None :
	"""
	Doc
	"""

	names = [
		'hidden_state_last', # output[0] =       tensor | last_hidden_state | batch_size, seq_len, hidden_size
		'features',          # output[1] =       tensor | pooler_output     | batch_size, hidden_size
		'hidden_state',      # output[3] = tuple tensor | hidden_states     | batch_size, seq_len, hidden_size      | config.output_hidden_states = True
		'attention'          # output[4] = tuple tensor | attentions        | batch_size, n_heads, seq_len, seq_len | config.output_attentions    = True
	]

	for index, output in enumerate(outputs) :
		process_extracion_result_single(
			inputs         = inputs,
			outputs        = output,
			directory      = directory,
			filename       = names[index],
			mode           = mode,
			should_evalute = should_evalute
		)

def process_extracion_result_single (inputs, outputs, directory : str, filename : str, mode : str = 'append', should_evalute : bool = False) -> None :
	"""
	Doc
	"""

	if should_evalute :
		filename = os.path.join(directory, 'dev_' + filename + '.txt')
	else :
		filename = os.path.join(directory, 'train_' + filename + '.txt')

	raise NotImplementedError()
