from Bio import SeqIO

from anndata import AnnData
from pandas  import DataFrame
from pyfaidx import Fasta
from typing  import Any
from typing  import Dict
from typing  import List

import anndata
import gff3_parser
import json
import numpy
import os
import pandas
import torch

from source.python.io._cleaner import clean_annotation
from source.python.io._cleaner import clean_metadata
from source.python.io._cleaner import clean_tpm

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

def load_json (filename : str) -> Dict[Any, Any] :
	"""
	Doc
	"""

	with open(filename, mode = 'r') as handle :
		data = json.load(handle)

	return data

def load_npz (filename : str, to_dict : bool = True) -> Dict[str, numpy.ndarray] | Any :
	"""
	Doc
	"""

	data = numpy.load(filename)

	if to_dict :
		data = dict(data.items())

	return data

def load_tsv (filename : str, low_memory : bool = True) -> DataFrame :
	"""
	Doc
	"""

	return pandas.read_csv(filename, low_memory = low_memory, sep = '\t')

def load_resources (directory : str, plant : str, clean : bool = False) -> Dict[str, Dict | DataFrame] :
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

def load_labels (filename : str, to_numpy : bool = False) -> Dict[str, Dict[str, List[float] | numpy.ndarray]] :
	"""
	Doc
	"""

	data = load_json(filename = filename)

	if to_numpy :
		for key, value in data.items() :
			data[key] = {k : numpy.array(v) for k, v in value.items()}

	return data
