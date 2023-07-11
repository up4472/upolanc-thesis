from pandas import DataFrame
from typing import Dict
from typing import List

def contains_any_keys (strings : List[str], key : str) -> bool :
	"""
	Doc
	"""

	for string in strings :
		if string in key :
			return True

	return False

def contains_all_keys (strings : List[str], key : str) -> bool :
	"""
	Doc
	"""

	for string in strings :
		if string not in key :
			return False

	return True

def filter_bert_reports_for (reports : Dict[str, DataFrame], keep_only : List[str] = None, drop_only : List[str] = None) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	if keep_only is not None and len(keep_only) == 0 : keep_only = None
	if drop_only is not None and len(drop_only) == 0 : drop_only = None

	keys = list(reports.keys())
	keep = [True for _ in keys]

	for index, key in enumerate(keys) :
		if keep_only is None : has_keep = True
		else                 : has_keep = contains_all_keys(keep_only, key)

		if drop_only is None : has_drop = False
		else                 : has_drop = contains_any_keys(drop_only, key)

		keep[index] = has_keep and not has_drop

	return {
		key : reports[key]
		for index, key in enumerate(keys)
		if keep[index]
	}

def filter_bert_reports (reports : Dict[str, Dict], keep_only : List[str] = None, drop_only : List[str] = None) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'regression'     : filter_bert_reports_for(reports = reports['regression'],     keep_only = keep_only, drop_only = drop_only),
		'classification' : filter_bert_reports_for(reports = reports['classification'], keep_only = keep_only, drop_only = drop_only)
	}
