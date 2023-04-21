from pandas import DataFrame
from typing import Dict
from typing import List

def filter_bert_reports_for (reports : Dict[str, DataFrame], keep_only : List[str] = None, drop_only : List[str] = None) -> Dict[str, DataFrame] :
	"""
	Doc
	"""

	if keep_only is not None and len(keep_only) == 0 : keep_only = None
	if drop_only is not None and len(drop_only) == 0 : drop_only = None

	keys = list(reports.keys())
	keep = [True for _ in keys]

	f1 = lambda x, y : any([i in y for i in x]) if x is not None else True
	f2 = lambda x, y : any([i in y for i in x]) if x is not None else False

	for index, key in enumerate(keys) :
		any_keep = f1(keep_only, key)
		any_drop = f2(drop_only, key)

		keep[index] = any_keep and not any_drop

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
