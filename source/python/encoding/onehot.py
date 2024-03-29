from Bio.Data import IUPACData

from typing import Dict
from typing import List
from typing import Union

import numpy

def generate_onehot_mapping (nucleotide_order : Union[str, List[str]] = None, ambiguous_value : str = 'zero') -> Dict[str, numpy.ndarray] :
	"""
	Doc
	"""

	data = dict()

	udna = IUPACData.unambiguous_dna_letters
	adna = IUPACData.ambiguous_dna_values

	if nucleotide_order and len(nucleotide_order) == 4 :
		udna = nucleotide_order

	for key, nucleotides in adna.items() :
		vector = numpy.zeros(shape = (4,), dtype = float)

		value = 1.0

		if key not in udna :
			ambiguous_value = ambiguous_value.lower()

			if   ambiguous_value == 'zero'     : value = 0.0
			elif ambiguous_value == 'one'      : value = 1.0
			elif ambiguous_value == 'fraction' : value = 1.0 / len(nucleotides)
			else : value = 0.0

		for nucleotide in nucleotides :
			vector[udna.index(nucleotide)] = value

		data[key] = vector

	return data

def onehot_encode (sequence : str, mapping : Dict[str, numpy.ndarray] = None, default : numpy.ndarray = None, transpose : bool = False) -> numpy.ndarray :
	"""
	Doc
	"""

	if default is None :
		default = numpy.zeros((4,), dtype = float)

	matrix = list()

	for base in sequence :
		if base in mapping :
			matrix.append(mapping[base])
		else :
			matrix.append(default)

	matrix = numpy.array(matrix)

	if transpose :
		return numpy.transpose(matrix)

	return matrix
