from Bio import SeqIO

from anndata import AnnData
from pandas  import DataFrame
from pyfaidx import Fasta
from typing  import Any
from typing  import Dict
from typing  import List
from typing  import Tuple
from typing  import Union

import anndata
import gff3_parser
import json
import numpy
import os
import pandas
import pickle
import torch

from sklearn.preprocessing import LabelBinarizer

from source.python.io.cleaner import clean_annotation
from source.python.io.cleaner import clean_metadata
from source.python.io.cleaner import clean_tpm
from source.python.io.writer  import write_pickle

def load_torch (filename : str) -> Dict[str, Any] :
	"""
	Doc
	"""

	return torch.load(filename)

def load_csv (filename : str, low_memory : bool = True) -> DataFrame :
	"""
	Doc
	"""

	return pandas.read_csv(filename, low_memory = low_memory)

def load_faidx (filename : str) -> Fasta :
	"""
	Doc
	"""

	return Fasta(filename)

def load_fasta (filename : str, to_string : bool = False) -> Dict[str, Any] :
	"""
	Doc
	"""

	data = dict()

	for record in SeqIO.parse(filename, 'fasta') :
		if to_string :
			data[record.id] = str(record.seq)
		else :
			data[record.id] = record

	return data

def load_gff3 (filename : str) -> DataFrame :
	"""
	Doc
	"""

	return gff3_parser.parse_gff3(filename,
		verbose          = False,
		parse_attributes = True
	)

def load_h5ad (filename : str) -> AnnData :
	"""
	Doc
	"""

	return anndata.read_h5ad(filename)

def load_json (filename : str) -> Union[List, Dict] :
	"""
	Doc
	"""

	with open(filename, mode = 'r') as handle :
		data = json.load(handle)

	return data

def load_npz (filename : str, to_dict : bool = True) -> Any :
	"""
	Doc
	"""

	data = numpy.load(filename)

	if to_dict :
		data = dict(data.items())

	return data

def load_parquet (filename : str) -> DataFrame :
	"""
	Doc
	"""

	return pandas.read_parquet(path = filename)

def load_pickle (filename : str) -> Any :
	"""
	Doc
	"""

	with open(filename, mode = 'rb') as handle :
		data = pickle.load(handle)

	return data

def load_tsv (filename : str, low_memory : bool = True) -> DataFrame :
	"""
	Doc
	"""

	return pandas.read_csv(filename, low_memory = low_memory, sep = '\t')

def load_resources (directory : str, plant : str, clean : bool = False) -> Dict[str, Any] :
	"""
	Doc
	"""

	directory = os.path.join(directory, plant)

	gene_annotation = load_gff3(
		filename = os.path.join(directory, 'gene-annotation.gff3')
	)

	gene_assembly = load_fasta(
		filename = os.path.join(directory, 'gene-assembly.fa')
	)

	tissue_metadata = load_tsv(
		filename = os.path.join(directory, 'tissue-metadata.tsv')
	)

	tissue_tpm = load_tsv(
		filename = os.path.join(directory, 'tissue-tpm.tsv')
	)

	if clean :
		gene_annotation = clean_annotation(dataframe = gene_annotation)
		tissue_metadata = clean_metadata(dataframe = tissue_metadata)
		tissue_tpm      = clean_tpm(dataframe = tissue_tpm)

	return {
		'gene_annotation' : gene_annotation,
		'gene_assembly'   : gene_assembly,
		'tissue_metadata' : tissue_metadata,
		'tissue_tpm'      : tissue_tpm
	}

def load_labels (filename : str, to_numpy : bool = False) -> Dict[str, Dict[str, Any]] :
	"""
	Doc
	"""

	data = load_json(filename = filename)

	if to_numpy :
		for key, value in data.items() :
			data[key] = {k : numpy.array(v) for k, v in value.items()}

	return data

def load_feature_targets (group : str, directory : str, filename : str, explode : bool = False, filters : Dict[str, Any] = None, mode : str = 'regression', cached : Any = None) -> Tuple[DataFrame, Dict, List] :
	"""
	Doc
	"""

	if cached is None :
		dataframe = load_pickle(filename = os.path.join(directory, filename))
	else :
		dataframe = cached

	dataframe = dataframe[group].set_index('ID')

	dataframe.index.name = None

	if explode :
		array = ['TPM_Value', 'TPM_Label']

		if 'Tissue'       in dataframe.columns : array.append('Tissue')
		if 'Age'          in dataframe.columns : array.append('Age')
		if 'Perturbation' in dataframe.columns : array.append('Perturbation')
		if 'Group'        in dataframe.columns : array.append('Group')
		if 'Global'       in dataframe.columns : array.append('Global')

		dataframe = dataframe.explode(array)

		dataframe.index = [
			'{}?{}'.format(g, i)
			for g, i in zip(
				dataframe[group.split('-')[0].capitalize()],
				dataframe.index
			)
		]

		for key in array :
			dataframe[key] = dataframe[key].apply(lambda x : [x])

	# ['t0', 't1', 't2'] ::  multi output :: keep target order
	# ['t0']             :: single output :: find target order

	target_order = dataframe[group.split('-')[0].capitalize()].iloc[0]

	if explode and len(target_order) == 1 :
		features_ext = dict()

		if filters[group.split('-')[0]] is None :
			target_order = ['mixed']
		else :
			target_order = filters[group.split('-')[0]]

		for column in dataframe.columns :
			if column not in ['Tissue', 'Group', 'Age', 'Perturbation'] :
				continue

			labels = [x[0] for x in dataframe[column]]
			binarizer = LabelBinarizer()
			binarizer = binarizer.fit(labels)
			encode = binarizer.transform(labels) # noqa

			for x, index in enumerate(dataframe.index) :
				group        = dataframe[column].iloc[x][0]
				group_filter = filters[column.lower()]

				if group_filter is None or group.lower() in group_filter :
					if index in features_ext.keys() :
						features_ext[index] = numpy.concatenate((features_ext[index], encode[x, :]))
					else :
						features_ext[index] = encode[x, :]

			write_pickle(
				data = binarizer,
				filename = os.path.join(directory, 'binarizer-{}.pkl'.format(column.lower()))
			)

		index = dataframe.index
		keys  = list(features_ext.keys())

		dataframe = dataframe.loc[index.isin(keys)].copy() # noqa

		features  = DataFrame.from_dict({k : [v] for k, v in features_ext.items()}, orient = 'index', columns = ['Feature'])
		dataframe = dataframe.merge(features, left_index = True, right_index = True)

	if mode == 'regression' :
		target_value = dataframe['TPM_Value'].to_dict()
		target_value = {k : numpy.array(v, dtype = numpy.float64) for k, v in target_value.items()}
	elif mode == 'classification' :
		target_value = dataframe['TPM_Label'].to_dict()
		target_value = {k : numpy.array(v, dtype = numpy.int64) for k, v in target_value.items()}
	else :
		raise ValueError()

	return dataframe, target_value, target_order

def load_model_configs (filename : str, sort : bool = False, sort_on : str = 'valid_r2', reverse : bool = False) -> List[Dict] :
	"""
	Doc
	"""

	if not os.path.exists(filename) :
		return []

	data = load_json(filename = filename)

	if isinstance(data, dict) :
		return [data]

	if sort :
		data = sorted(data, key = lambda key : key[sort_on], reverse = reverse)

	return data
