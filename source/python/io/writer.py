from anndata import AnnData
from os      import PathLike
from pandas  import DataFrame
from pathlib import Path
from typing  import Any
from typing  import Dict

import json
import numpy
import pickle

def write_csv (data : DataFrame, filename : str, write_index : bool = False) -> None :
	"""
	Doc
	"""

	data.to_csv(filename, index = write_index, sep = ',')

def write_fasta (data : Dict[str, Any], filename : str) -> None :
	"""
	Doc
	"""

	with open(filename, mode = 'w') as handle :
		for key, record in data.items() :
			handle.write('>')
			handle.write(str(key))
			handle.write('\n')
			handle.write(str(record))
			handle.write('\n')

def write_h5ad (data : AnnData, filename : str) -> None :
	"""
	Doc
	"""

	data.write_h5ad(filename = Path(filename))

def write_json (data : Dict[Any, Any], filename : PathLike) -> None :
	"""
	Doc
	"""

	with open(filename, mode = 'w') as handle :
		json.dump(data, handle, indent = '\t', separators = (',', ' : '))

def write_npz (data : Dict[Any, Any], filename : str) -> None :
	"""
	Doc
	"""

	numpy.savez(filename, **data)

def write_parquet (data : DataFrame, filename : str) -> None :
	"""
	Doc
	"""

	data.to_parquet(path = filename)

def write_pickle (data : Any, filename : str) -> None :
	"""
	Doc
	"""

	with open(filename, mode = 'wb') as handle :
		pickle.dump(data, handle)

def write_tsv (data : DataFrame, filename : str, write_index : bool = False) -> None :
	"""
	Doc
	"""

	data.to_csv(filename, index = write_index, sep = '\t')
