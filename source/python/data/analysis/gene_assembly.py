from Bio.Data      import IUPACData
from Bio.SeqRecord import SeqRecord

from pandas import DataFrame
from typing import Dict

def show (data : Dict[str, SeqRecord], head : int = 10, tail : int = 10) -> None :
	"""
	Doc
	"""

	head_length = head
	tail_length = tail

	for record in data.values() :
		key = str(record.id)
		seq = str(record.seq)

		head = seq[:head_length]
		tail = seq[-tail_length:]

		print(f'Sequence [{key:2s}] with length [{len(seq):>10,d}] : {head} ... {tail}')

def show_nucleotide_frequency (data : Dict[str, SeqRecord], relative : bool = True) -> DataFrame :
	"""
	Doc
	"""

	nucleotides = list(IUPACData.ambiguous_dna_values.keys())
	frequencies = dict()

	for name, record in data.items() :
		frequency = {nucleotide : 0 for nucleotide in nucleotides}

		for nucleotide in str(record.seq) :
			frequency[nucleotide] = frequency[nucleotide] + 1

		frequencies[name] = frequency

	frequencies['Total'] = {
		nucleotide : sum(
			frequency[nucleotide]
				for frequency in frequencies.values()
		)
		for nucleotide in nucleotides
	}

	if relative :
		for key, frequency in frequencies.items() :
			total_nucleotides = sum(frequency.values())

			for nucleotide, count in frequency.items() :
				frequency[nucleotide] = count / total_nucleotides

			frequencies[key] = frequency

	dataframe = DataFrame.from_dict(frequencies, orient = 'index')
	dataframe = dataframe.loc[:, (dataframe != 0).any(axis = 0)]

	return dataframe.sort_values(by = 'Total', ascending = False, axis = 1)
