from typing import Callable
from typing import Dict

def get_subsequences (sequences : Dict[str, str], start : int = None, end : int = None) -> Dict[str, str] :
	"""
	Doc
	"""

	if start is None and end is None :
		return sequences

	if start is None : start = 0
	if end   is None : end   = 2150

	return {
		key : value[start:end]
		for key, value in sequences.items()
	}

def get_encoding (sequences : Dict[str, str], should_encode : bool, expand_dims : int, encoder : Callable, expander : Callable) -> Dict[str, str] :
	"""
	Doc
	"""

	if not should_encode :
		return sequences

	sequences = {
		key : encoder(value)
		for key, value in sequences.items()
	}

	if expand_dims is not None and expand_dims >= 0 :
		sequences = {
			key : expander(value)
			for key, value in sequences.items()
		}

	return sequences
