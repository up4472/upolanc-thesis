from Bio.Data import IUPACData
from typing   import List
from typing   import Union

import itertools
import numpy
import textwrap

def gc_content (sequence : str) -> float :
	"""
	Doc
	"""

	if len(sequence) > 0 :
		g = sequence.count('G')
		c = sequence.count('C')

		return (g + c) / len(sequence)

	return 0.0

def gc_count (sequence : str) -> List[float] :
	"""
	Doc
	"""

	gc1 = 0
	gc2 = 0
	gc3 = 0

	for codon in textwrap.wrap(sequence, width = 3) :
		if len(codon) != 3 :
			continue

		if codon[0] in ['G', 'C'] : gc1 = gc1 + 1
		if codon[1] in ['G', 'C'] : gc2 = gc2 + 1
		if codon[2] in ['G', 'C'] : gc3 = gc3 + 1

	gc1 = 3.0 * gc1 / len(sequence)
	gc2 = 3.0 * gc2 / len(sequence)
	gc3 = 3.0 * gc3 / len(sequence)

	return [gc1, gc2, gc3]

def codon_frequency (sequence : str, mrna : str, relative : bool = True) -> List[Union[int, float]] :
	"""
	Doc
	"""

	nucleotides = IUPACData.unambiguous_dna_letters

	codons = [''.join(codon) for codon in itertools.product(nucleotides, repeat = 3)]
	codons = {codon : 0 for codon in codons}

	for codon in textwrap.wrap(sequence, width = 3) :
		if len(codon) != 3 :
			continue

		if codon in codons :
			codons[codon] = codons[codon] + 1
		else :
			print('[{:12s}]'.format(mrna) + f' : unknown codon [{codon}]')

	if relative :
		total_count = sum(codons.values())

		for codon, count in codons.items() :
			codons[codon] = count / total_count

	keys = codons.keys()
	keys = sorted(keys)

	return [codons[key] for key in keys]

def mrna_stability (utr5 : str, utr3 : str, cds : str, utr5_length : int, utr3_length : int, cds_length : int) -> List[Union[int, float]] :
	"""
	Doc
	"""

	gc_utr5 = gc_content(sequence = utr5)
	gc_utr3 = gc_content(sequence = utr3)
	gc_cds  = gc_count(sequence = cds)

	return [
		utr5_length / float(300),
		utr3_length / float(350),
		cds_length / float(1500),
		gc_utr5,
		gc_utr3,
		gc_cds[0],
		gc_cds[1],
		gc_cds[2]
	]
