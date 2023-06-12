from Bio.Data import IUPACData
from Bio.Seq  import Seq

from pandas  import DataFrame
from pyfaidx import Fasta
from pyfaidx import FastaRecord
from typing  import Dict
from typing  import List
from typing  import Tuple
from typing  import Union

import itertools
import numpy
import pandas
import textwrap

from source.python.data.feature.feature_methods import codon_frequency
from source.python.data.feature.feature_methods import mrna_stability

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

	data = DataFrame(columns = ['Gene', 'Transcript', 'Length', 'Index'])
	mrna = dataframe[dataframe['Type'] == 'mRNA']

	data['Gene']       = mrna['Gene']
	data['Transcript'] = mrna['Transcript']
	data['Length']     = numpy.absolute(mrna['Start'] - mrna['End'])
	data['Index']      = data.index

	data = data.reset_index(drop = True)

	return data

def group_regions_list (dataframe : DataFrame, region : str) -> DataFrame :
	"""
	Doc
	"""

	tmp = dataframe.assign(Limits = dataframe[dataframe['Type'] == region][['Start', 'End']].apply(lambda x : x.values, axis = 1))
	out = tmp[tmp['Type'] == region].groupby('Transcript', as_index = False)['Limits'].agg(list)

	return out

def group_regions_with_merge (dataframe : DataFrame, annotation : DataFrame, region : str) -> DataFrame :
	"""
	Doc
	"""

	regions = group_regions_list(dataframe = annotation, region = region)
	regions = regions.loc[regions['Transcript'].isin(dataframe['Transcript'])]

	dataframe = pandas.merge(dataframe, regions, on = 'Transcript')
	dataframe[region].update(dataframe['Limits'])
	dataframe.drop(columns = ['Limits'], inplace = True)

	return dataframe

def annotation_to_regions (annotation : DataFrame, lengths : Dict[str, Union[int, List[int]]], verbose : bool = True) -> DataFrame :
	"""
	Doc
	"""

	LEN_PROM_FULL = lengths['prom_full']
	LEN_PROM      = lengths['prom']
	LEN_TERM      = lengths['term']
	LEN_TERM_FULL = lengths['term_full']

	annotation = annotation[annotation['Type'].isin(['mRNA', 'UTR5', 'CDS', 'UTR3'])].copy()

	select = extract_longest_names(dataframe = annotation)
	dataframe = DataFrame(columns = ['Gene', 'Transcript', 'CDS', 'UTR5', 'UTR3', 'Start', 'End', 'Strand', 'Seq'])

	annotation = annotation[annotation['Transcript'].isin(select['Transcript'])]

	dataframe['Start']  = annotation[annotation['Type'] == 'mRNA']['Start']
	dataframe['End']    = annotation[annotation['Type'] == 'mRNA']['End']
	dataframe['Strand'] = annotation[annotation['Type'] == 'mRNA']['Strand']
	dataframe['Seq']    = annotation[annotation['Type'] == 'mRNA']['Seq']

	dataframe['Transcript'] = annotation[annotation['Type'] == 'mRNA']['Transcript']
	dataframe['Gene']       = select.set_index(['Transcript']).loc[dataframe['Transcript'].values]['Gene'].values

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
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '+', 'Prom'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts
	e = tts + LEN_TERM - 1
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '+', 'Term'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tss - LEN_PROM_FULL[0] + 1
	e = tss + LEN_PROM_FULL[1]
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '+', 'Prom_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts - LEN_TERM_FULL[0]
	e = tts + LEN_TERM_FULL[1] - 1
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '+', 'Term_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	tss = dataframe[dataframe['Strand'] == '-']['Start']
	tts = dataframe[dataframe['Strand'] == '-']['End']

	tss = tss + 1
	tts = tts - 1

	s = tss - LEN_TERM + 1
	e = tss
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '-', 'Term'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts
	e = tts + LEN_PROM - 1
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '-', 'Prom'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tss - LEN_TERM_FULL[1] + 1
	e = tss + LEN_TERM_FULL[0]
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '-', 'Term_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	s = tts - LEN_PROM_FULL[1]
	e = tts + LEN_PROM_FULL[0] - 1
	s[s < 1] = 1
	e[e < 1] = 1
	dataframe.loc[dataframe['Strand'] == '-', 'Prom_Full'] = pandas.concat([s, e], axis = 1).apply(lambda1, axis = 1).apply(lambda2)

	dataframe['CDS_Length']  = dataframe[ 'CDS'].apply(lambda x : 0 if numpy.any(pandas.isna(x)) else sum(numpy.diff(x))[0])
	dataframe['UTR5_Length'] = dataframe['UTR5'].apply(lambda x : 0 if numpy.any(pandas.isna(x)) else sum(numpy.diff(x))[0])
	dataframe['UTR3_Length'] = dataframe['UTR3'].apply(lambda x : 0 if numpy.any(pandas.isna(x)) else sum(numpy.diff(x))[0])

	dataframe = dataframe[
		(dataframe[ 'CDS_Length'] < 20_000) &
		(dataframe['UTR5_Length'] < 10_000) &
		(dataframe['UTR3_Length'] < 10_000)
	].reset_index(drop = True)

	if verbose :
		print('Passed 1st assertion : ' + str(len(dataframe['Transcript'].unique()) == len(dataframe['Transcript'])))
		print('Passed 2nd assertion : ' + str(not dataframe.isnull().values.any()))

	return dataframe[[
		'Seq', 'Strand', 'Gene', 'Transcript', 'Start', 'End',
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

def regions_to_features (faidx : Fasta, dataframe : DataFrame, lengths : Dict[str, Union[int, List[int]]], verbose : bool = True) -> Tuple[DataFrame, DataFrame] :
	"""
	Doc
	"""

	LEN_PROM = lengths['prom']
	LEN_UTR5 = lengths['utr5']
	LEN_UTR3 = lengths['utr3']
	LEN_TERM = lengths['term']

	dataframe = dataframe.copy(deep = True)
	dataframe.reset_index(drop = True, inplace = True)

	seqcols = ['Gene', 'Transcript', 'Prom_UTR5', 'Prom_Full', 'Prom', 'UTR5', 'CDS', 'UTR3', 'Term', 'Term_Full']
	varcols = ['Gene', 'Transcript', 'Frequency', 'Stability']

	sequences = DataFrame(columns = seqcols)
	variables = DataFrame(columns = varcols)

	for index, row in dataframe.iterrows() :
		seq = str(row['Seq'])

		if seq.lower() in ['mt'] :
			continue

		utr_min = int(dataframe.at[index, 'UTR_Min'])
		utr_max = int(dataframe.at[index, 'UTR_Max'])

		if dataframe.at[index, 'Strand'] == '+' :
			xmin = utr_min - LEN_PROM
			xmax = utr_max + LEN_TERM
		else :
			xmin = utr_min - LEN_TERM
			xmax = utr_max + LEN_PROM

		if xmin <= 0 or xmax > len(faidx[seq]) :
			if verbose :
				print('[{:12s}] : out of bounds | [{:7d}] [{:7d}] for [{:7d}] [{:7d}]'.format(
					row['Transcript'], xmin, xmax, 0, len(faidx[seq]))
				)

			continue

		sequences.at[index, 'Gene'] = row['Gene']
		variables.at[index, 'Gene'] = row['Gene']

		sequences.at[index, 'Transcript'] = row['Transcript']
		variables.at[index, 'Transcript'] = row['Transcript']

		for name in ['Prom_UTR5', 'Prom_Full', 'Prom', 'CDS', 'Term', 'Term_Full'] :
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
			mrna     = row['Transcript'],
			relative = True,
			verbose  = verbose
		)

		variables.at[index, 'Stability'] = mrna_stability(
			utr5        = utr5,
			utr3        = utr3,
			cds         = sequences.at[index, 'CDS'],
			utr5_length = row['UTR5_Length'],
			utr3_length = row['UTR3_Length'],
			cds_length  = row['CDS_Length']
		)

	sequences = sequences.dropna().reset_index(drop = True)
	variables = variables.dropna(subset = ['Transcript']).reset_index(drop = True)

	return sequences, variables

def sequences_extend_kvpair (sequences : Dict[str, Dict], regions : DataFrame, header : str = None) -> Dict[str, Dict] :
	"""
	Doc
	"""

	fcount = 0 if header is None else header.count('{}')

	data = dict()

	for mrna, region_dict in sequences.items() :
		if mrna not in data.keys() :
			data[mrna] = dict()

		region = regions.loc[regions['Transcript'] == mrna]

		strand = str(region['Strand'].iloc[0])
		seqid  = str(region['Seq'].iloc[0])

		for region_key, region_seq in region_dict.items() :
			if region_key not in ['Prom_UTR5', 'Prom_Full', 'Prom', 'UTR5', 'CDS', 'UTR3', 'Term', 'Term_Full'] :
				continue

			x = -1
			y = -1

			for limit in region[region_key].iloc[0] :
				imin = min(*limit)
				imax = max(*limit)

				x = imin if x < 0 else min(x, imin)
				y = imax if y < 0 else max(y, imax)

			key = mrna

			if   fcount == 1 : key = header.format(mrna)
			elif fcount == 2 : key = header.format(mrna, strand)
			elif fcount == 3 : key = header.format(mrna, strand, seqid)
			elif fcount == 5 : key = header.format(mrna, strand, seqid, x, y)
			elif fcount == 6 : key = header.format(mrna, strand, seqid, x, y, len(region_seq))

			data[mrna].update({
				region_key : {
					'key' : key,
					'seq' : region_seq
				}
			})

	return data

def print_extracted_sequence (transcript : str, sequences : Dict[str, Dict], width : int = 10, columns : int = 10, space : bool = True) -> None :
	"""
	Doc
	"""

	for r in ['Prom_UTR5', 'Prom_Full', 'Prom', 'UTR5', 'CDS', 'UTR3', 'Term', 'Term_Full'] :
		key = sequences[transcript][r]['key']
		seq = sequences[transcript][r]['seq']

		print(key)

		for index, subseq in enumerate(textwrap.wrap(seq, width = width)) :
			if index > 0 and index % columns == 0 : print()
			print(subseq, end = ' ' if space else '')

		print()
		print()

def print_padded_sequence (transcript : str, sequences : Dict[str, str], width : int = 10, columns : int = 10, space : bool = True) -> None :
	"""
	Doc
	"""

	mapping = {
		key.split(' | ')[0] : key
		for key in sequences.keys()
	}

	key = mapping[transcript]
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
		side = side.lower()

		if   side == 'left'  : side = -1
		elif side == 'none'  : side =  0
		elif side == 'right' : side =  1
		else : raise ValueError()

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

def merge_and_pad_sequences (sequences : Dict[str, Dict], lengths : Dict[str, Union[int, List[int]]], padding : Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]] :
	"""
	Doc
	"""

	data_large = dict()
	data_small = dict()

	for key, value in sequences.items() :
		prom_full = pad_single(
			sequence  = value['Prom_Full']['seq'],
			side      = padding['prom_full'],
			length    = lengths['prom_full'][0] + lengths['prom_full'][1],
			pad_value = None
		)

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

		term_full = pad_single(
			sequence  = value['Term_Full']['seq'],
			side      = padding['term_full'],
			length    = lengths['term_full'][0] + lengths['term_full'][1],
			pad_value = None
		)

		strand = value['CDS']['key'].split(' | ')[1]
		seqid  = value['CDS']['key'].split(' | ')[2].split(':')[0]

		if strand == '+' :
			large_0 = value['Prom_Full']['key'].split(' | ')[2].split(':')[1].split('-')[0]
			large_1 = value['Term_Full']['key'].split(' | ')[2].split(':')[1].split('-')[1]

			small_0 = value['Prom']['key'].split(' | ')[2].split(':')[1].split('-')[0]
			small_1 = value['Term']['key'].split(' | ')[2].split(':')[1].split('-')[1]
		else :
			large_0 = value['Term_Full']['key'].split(' | ')[2].split(':')[1].split('-')[0]
			large_1 = value['Prom_Full']['key'].split(' | ')[2].split(':')[1].split('-')[1]

			small_0 = value['Term']['key'].split(' | ')[2].split(':')[1].split('-')[0]
			small_1 = value['Prom']['key'].split(' | ')[2].split(':')[1].split('-')[1]

		seq_small = prom + utr5 + utr3 + term
		key_small = '{} | {} | {}:{}-{} | {}'.format(key, strand, seqid, small_0, small_1, len(seq_small))

		data_small[key_small] = seq_small

		seq_large = prom_full + utr5 + utr3 + term_full
		key_large = '{} | {} | {}:{}-{} | {}'.format(key, strand, seqid, large_0, large_1, len(seq_large))

		data_large[key_large] = seq_large

	return data_small, data_large
