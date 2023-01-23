from Bio.Data import IUPACData
from Bio.Seq  import Seq

from pandas  import DataFrame
from pyfaidx import Fasta
from pyfaidx import FastaRecord
from typing  import Dict
from typing  import List
from typing  import Tuple
from typing import Union

from tqdm.notebook import tqdm

import itertools
import numpy
import pandas
import textwrap

def compute_codon_frequency (sequence : str, relative : bool = True, as_array : bool = False) -> Union[Dict[str, float], numpy.ndarray] :
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

	if relative :
		total_count = sum(codons.values())

		for codon, count in codons.items() :
			codons[codon] = count / total_count

	if as_array :
		keys = codons.keys()
		keys = sorted(keys)

		codons = numpy.array([codons[key] for key in keys])

	return codons

def extract_longest_names (dataframe : DataFrame) -> DataFrame :
	"""
	Doc
	"""

	data = DataFrame(columns = ['Gene', 'mRNA', 'Length', 'Index'])
	mrna = dataframe[dataframe['Type'] == 'mRNA']

	data['Gene']   = mrna['Gene']
	data['mRNA']   = mrna['mRNA']
	data['Length'] = numpy.absolute(mrna['Start'] - mrna['End'])
	data['Index']  = data.index

	data = data.reset_index(drop = True)

	return data

def group_regions_list (dataframe : DataFrame, region : str) -> DataFrame :
	"""
	Doc
	"""

	tmp = dataframe.assign(Limits = dataframe[dataframe['Type'] == region][['Start', 'End']].apply(lambda x : x.values, axis = 1))
	out = tmp[tmp['Type'] == region].groupby('mRNA', as_index = False)['Limits'].agg(list)

	return out

def group_regions_with_merge (dataframe : DataFrame, annotation : DataFrame, region : str) -> DataFrame :
	"""
	Doc
	"""

	regions = group_regions_list(dataframe = annotation, region = region)
	regions = regions.loc[regions['mRNA'].isin(dataframe['mRNA'])]

	dataframe = pandas.merge(dataframe, regions, on = 'mRNA')
	dataframe[region].update(dataframe['Limits'])
	dataframe.drop(columns = ['Limits'], inplace = True)

	return dataframe

def annotation_to_regions (annotation : DataFrame, lengths : Dict[str, Union[int, List[int]]]) -> DataFrame :
	"""
	Doc
	"""

	LEN_PROM_FULL = lengths['prom_full']
	LEN_PROM      = lengths['prom']
	LEN_TERM      = lengths['term']
	LEN_TERM_FULL = lengths['term_full']

	annotation = annotation[annotation['Type'].isin(['Gene', 'mRNA', 'UTR5', 'Exon', 'CDS', 'UTR3'])].copy()

	select = extract_longest_names(dataframe = annotation)
	dataframe = DataFrame(columns = ['Gene', 'mRNA', 'CDS', 'UTR5', 'UTR3', 'Start', 'End', 'Strand', 'Seq'])

	annotation = annotation[annotation['mRNA'].isin(select['mRNA'])]

	dataframe['Start']  = annotation[annotation['Type'] == 'mRNA']['Start']
	dataframe['End']    = annotation[annotation['Type'] == 'mRNA']['End']
	dataframe['Strand'] = annotation[annotation['Type'] == 'mRNA']['Strand']
	dataframe['Seq']    = annotation[annotation['Type'] == 'mRNA']['Seq']

	dataframe['mRNA'] = annotation[annotation['Type'] == 'mRNA']['mRNA']
	dataframe['Gene'] = select.set_index(['mRNA']).loc[dataframe['mRNA'].values]['Gene'].values

	dataframe = group_regions_with_merge(dataframe = dataframe, annotation = annotation, region = 'CDS')
	dataframe = group_regions_with_merge(dataframe = dataframe, annotation = annotation, region = 'UTR5')
	dataframe = group_regions_with_merge(dataframe = dataframe, annotation = annotation, region = 'UTR3')

	dataframe.dropna(subset = ['CDS', 'UTR5', 'UTR3', 'Start', 'End'], inplace = True)

	lambda1 = lambda x : x.values
	lambda2 = lambda x : [x]

	tss = dataframe[dataframe['Strand'] == '+']['Start']
	tts = dataframe[dataframe['Strand'] == '+']['End']

	tss = tss - 1
	tts = tts + 1

	s = tss - LEN_PROM + 1
	e = tss
	dataframe.loc[dataframe['Strand'] == '+', 'Prom'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts
	e = tts + LEN_TERM - 1
	dataframe.loc[dataframe['Strand'] == '+', 'Term'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tss - LEN_PROM_FULL[0] + 1
	e = tss + LEN_PROM_FULL[1]
	dataframe.loc[dataframe['Strand'] == '+', 'Prom_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts - LEN_TERM_FULL[0]
	e = tts + LEN_TERM_FULL[1] - 1
	dataframe.loc[dataframe['Strand'] == '+', 'Term_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	tss = dataframe[dataframe['Strand'] == '-']['Start']
	tts = dataframe[dataframe['Strand'] == '-']['End']

	tss = tss + 1
	tts = tts - 1

	s = tss - LEN_TERM + 1
	e = tss
	dataframe.loc[dataframe['Strand'] == '-', 'Term'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts
	e = tts + LEN_PROM - 1
	dataframe.loc[dataframe['Strand'] == '-', 'Prom'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tss - LEN_TERM_FULL[1] + 1
	e = tss + LEN_TERM_FULL[0]
	dataframe.loc[dataframe['Strand'] == '-', 'Term_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts - LEN_PROM_FULL[1]
	e = tts + LEN_PROM_FULL[0] - 1
	dataframe.loc[dataframe['Strand'] == '-', 'Prom_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	dataframe['CDS_Length']  = dataframe[ 'CDS'].apply(lambda x : 0 if numpy.any(pandas.isna(x)) else sum(numpy.diff(x))[0])
	dataframe['UTR5_Length'] = dataframe['UTR5'].apply(lambda x : 0 if numpy.any(pandas.isna(x)) else sum(numpy.diff(x))[0])
	dataframe['UTR3_Length'] = dataframe['UTR3'].apply(lambda x : 0 if numpy.any(pandas.isna(x)) else sum(numpy.diff(x))[0])

	dataframe = dataframe[
		(dataframe[ 'CDS_Length'] < 20_000) &
		(dataframe['UTR5_Length'] < 10_000) &
		(dataframe['UTR3_Length'] < 10_000)
	].reset_index(drop = True)

	print('Passed 1st assertion : ' + str(len(dataframe['mRNA'].unique()) == len(dataframe['mRNA'])))
	print('Passed 2nd assertion : ' + str(not dataframe.isnull().values.any()))

	return dataframe[[
		'Seq', 'Strand', 'Gene', 'mRNA', 'Start', 'End',
		'Prom_Full', 'Prom', 'UTR5', 'CDS', 'UTR3', 'Term', 'Term_Full',
		'UTR5_Length', 'CDS_Length', 'UTR3_Length'
	]]

def extract_region_from_list (chromosome : FastaRecord, region : List[List[int]], strand : str) -> str :
	"""
	Doc
	"""

	sequence = list()

	if not numpy.any(pandas.isna(region)) and not not region[0][0] :
		for segment in region :
			s = int(segment[0])
			e = int(segment[1])

			if e - s > 0 :
				sequence.append(str(chromosome[s - 1:e]))
			else :
				sequence.append('')

		sequence = ''.join(sequence)

		if strand == '-' :
			sequence = Seq(sequence)
			sequence = sequence.reverse_complement()
			sequence = str(sequence)

		return sequence

	return ''

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

def regions_to_features (faidx : Fasta, dataframe : DataFrame, lengths : Dict[str, Union[int, List[int]]]) -> Tuple[DataFrame, DataFrame] :
	"""
	Doc
	"""

	LEN_PROM      = lengths['prom']
	LEN_UTR5      = lengths['utr5']
	LEN_UTR3      = lengths['utr3']
	LEN_TERM      = lengths['term']

	dataframe = dataframe.copy(deep = True)
	dataframe.reset_index(drop = True, inplace = True)

	seqcols = ['Gene', 'mRNA', 'Prom_Full', 'Prom', 'UTR5', 'CDS', 'UTR3', 'Term', 'Term_Full']
	varcols = ['Gene', 'mRNA', 'Frequency', 'Stability']

	sequences = DataFrame(columns = seqcols)
	variables = DataFrame(columns = varcols)

	for index, row in tqdm(dataframe.iterrows(), total = len(dataframe)) :
		seq = str(row['Seq'])

		if seq.lower() in ['mt'] :
			continue

		if dataframe.at[index, 'Strand'] == '+' :
			start = int(dataframe.at[index, 'Start'])
			end   = int(dataframe.at[index, 'End'])

			if start - LEN_PROM < 0 or end + LEN_TERM > len(faidx[seq]) :
				print('[{:12s}]'.format(row['mRNA']) + ' : out of bounds at sequence start')
				continue
		else :
			start = int(dataframe.at[index, 'End'])
			end   = int(dataframe.at[index, 'Start'])

			if start + LEN_PROM > len(faidx[seq]) or end - LEN_TERM < 0 :
				print('[{:12s}]'.format(row['mRNA']) + ' : out of bounds at sequence end')
				continue

		sequences.at[index, 'Gene'] = row['Gene']
		variables.at[index, 'Gene'] = row['Gene']

		sequences.at[index, 'mRNA'] = row['mRNA']
		variables.at[index, 'mRNA'] = row['mRNA']

		for name in ['Prom_Full', 'Prom', 'CDS', 'Term', 'Term_Full'] :
			sequences.at[index, name] = extract_region_from_list(
				chromosome = faidx[seq],
				region     = row[name],
				strand     = row['Strand']
			)

		utr5 = extract_region_from_list(chromosome = faidx[seq], region = row['UTR5'], strand = row['Strand'])
		utr3 = extract_region_from_list(chromosome = faidx[seq], region = row['UTR3'], strand = row['Strand'])

		sequences.at[index, 'UTR5'] = utr5[-LEN_UTR5:]
		sequences.at[index, 'UTR3'] = utr3[-LEN_UTR3:]

		variables.at[index, 'Frequency'] = codon_frequency(
			sequence = sequences.at[index, 'CDS'],
			mrna     = row['mRNA'],
			relative = True
		)

		gc_utr5 = gc_content(sequence = utr5)
		gc_utr3 = gc_content(sequence = utr3)
		gc_cds  = gc_count(sequence = sequences.at[index, 'CDS'])

		variables.at[index, 'Stability'] = [
			row['UTR5_Length'],
			row['UTR3_Length'],
			row['CDS_Length'],
			1000 * gc_utr5,
			1000 * gc_utr3,
			1000 * gc_cds[0],
			1000 * gc_cds[1],
			1000 * gc_cds[2]
		]

	sequences = sequences.dropna().reset_index(drop = True)
	variables = variables.dropna(subset = ['mRNA']).reset_index(drop = True)

	return sequences, variables

def sequences_extend_kvpair (sequences : Dict[str, Dict], regions : DataFrame, header : str) -> Dict[str, Dict] :
	"""
	Doc
	"""

	data = dict()

	for mrna, region_dict in sequences.items() :
		if mrna not in data.keys() :
			data[mrna] = dict()

		region = regions.loc[regions['mRNA'] == mrna]

		strand = str(region['Strand'].iloc[0])
		seqid  = str(region['Seq'].iloc[0])

		for region_key, region_seq in region_dict.items() :
			if region_key not in ['Prom_Full', 'Prom', 'UTR5', 'CDS', 'UTR3', 'Term', 'Term_Full'] :
				continue

			x = -1
			y = -1

			for limit in region[region_key].iloc[0] :
				imin = min(*limit)
				imax = max(*limit)

				x = imin if x < 0 else min(x, imin)
				y = imax if y < 0 else max(y, imax)

			data[mrna].update({
				region_key : {
					'key' : header.format(mrna, strand, seqid, x, y, len(region_seq)),
					'seq' : region_seq
				}
			})

	return data

def print_extracted_sequence (mrna : str, sequences : Dict[str, Dict], width : int = 10, columns : int = 10, space : bool = True) -> None :
	"""
	Doc
	"""

	for r in ['Prom', 'UTR5', 'CDS', 'UTR3', 'Term'] :
		key = sequences[mrna][r]['key']
		seq = sequences[mrna][r]['seq']

		print(key)

		for index, subseq in enumerate(textwrap.wrap(seq, width = width)) :
			if index > 0 and index % columns == 0 : print()
			print(subseq, end = ' ' if space else '')

		print()
		print()

def print_padded_sequence (mrna : str, sequences : Dict[str, str], width : int = 10, columns : int = 10, space : bool = True) -> None :
	"""
	Doc
	"""

	mapping = {
		key.split(' | ')[0] : key
		for key in sequences.keys()
	}

	key = mapping[mrna]
	seq = sequences[key]

	print(key)

	for index, subseq in enumerate(textwrap.wrap(seq, width = width)) :
		if index > 0 and index % columns == 0 : print()
		print(subseq, end = ' ' if space else '')

	print()
	print()

def pad_single (sequence : str, length : Union[int, List[int]], side : Union[int, str], pad_value : str = None) -> str :
	"""
	Doc
	"""

	if isinstance(length, list) :
		length = sum(length)

	if pad_value is None :
		pad_value = '-'

	if isinstance(side, str) :
		match side.lower() :
			case 'left'  : side = -1
			case 'none'  : side =  0
			case 'right' : side =  1
			case _       : raise ValueError()

	diff = length - len(sequence)

	if diff == 0 or side == 0 :
		return sequence

	if diff < 0 :
		print(f'Warning : {length} < {len(sequence)}')

	if side < 0 :
		return pad_value * diff + sequence

	return sequence + pad_value * diff

def pad_multiple (sequences : Dict[str, str], length : Union[int, List[int]], side : Union[int, str], pad_value : str = None) -> Dict[str, str] :
	"""
	Doc
	"""

	return {
		key : pad_single(
			sequence  = value,
			length    = length,
			side      = side,
			pad_value = pad_value
		)
		for key, value in sequences.items()
	}

def merge_and_pad_sequences (sequences : Dict[str, Dict], lengths : Dict[str, Union[int, List[int]]], padding : Dict[str, str]) -> Dict[str, str] :
	"""
	Doc
	"""

	data = dict()

	for key, value in sequences.items() :
		prom = pad_single(
			sequence  = value['Prom']['seq'],
			side      = padding['prom'],
			length    = lengths['prom'],
			pad_value = None
		)

		utr5 = pad_single(
			sequence  = value['UTR5']['seq'],
			side      = padding['utr5'],
			length    = lengths['utr5'],
			pad_value = None
		)

		utr3 = pad_single(
			sequence  = value['UTR3']['seq'],
			side      = padding['utr3'],
			length    = lengths['utr3'],
			pad_value = None
		)

		term = pad_single(
			sequence  = value['Term']['seq'],
			side      = padding['term'],
			length    = lengths['term'],
			pad_value = None
		)

		strand = value['CDS']['key'].split(' | ')[1]
		seqid  = value['CDS']['key'].split(' | ')[2].split(':')[0]

		if strand == '+' :
			start = value['Prom']['key'].split(' | ')[2].split(':')[1].split('-')[0]
			end   = value['Term']['key'].split(' | ')[2].split(':')[1].split('-')[1]
		else :
			start = value['Term']['key'].split(' | ')[2].split(':')[1].split('-')[0]
			end   = value['Prom']['key'].split(' | ')[2].split(':')[1].split('-')[1]

		seq = prom + utr5 + utr3 + term
		key = '{} | {} | {}:{}-{} | {}'.format(key, strand, seqid, start, end, len(seq))

		data[key] = seq

	return data
