from collections import Counter
from typing      import Dict
from typing      import List
from typing      import Tuple

from tqdm.auto import tqdm

import itertools

from source.python.data.feature.feature import codon_frequency
from source.python.data.synth._mutation import mutate_exponential
from source.python.data.synth._mutation import mutate_random

def mutate_sequences (sequences : Dict[str, Dict], rates : List[float], params : Dict[str, float], variants : int, method : str) -> Tuple[Dict, Dict] :
	"""
	Doc
	"""

	m_sequences = dict()
	m_features  = dict()

	transcripts = list(sequences.keys())
	regions     = list(sequences[transcripts[0]].keys())

	total = len(transcripts) * len(rates)

	for transcript, rate in tqdm(itertools.product(transcripts, rates), total = total) :
		params['mutation_rate'] = rate

		# Keep orignal as well
		baseline = transcript + '-M00.0'

		m_sequences[baseline] = sequences[transcript]
		m_features[baseline] = {
			'Frequency' : codon_frequency(
				sequence = m_sequences[baseline]['CDS']['seq'],
				mrna     = baseline,
				relative = True
			),
			'Stability' : [0] * 8
		}

		for variant in range(variants) :
			key = '{}-M{:02.0f}.{}'.format(transcript, 100 * rate, variant)

			m_sequences[key] = dict()
			m_features[key]  = dict()

			for region in regions :
				m_sequences[key][region] = {
					'seq' : mutate_sequence(
						sequence       = sequences[transcript][region]['seq'],
						mutation_rates = params,
						method         = method
					)[0],
					'key' : sequences[transcript][region]['key']
				}

			m_features[key] = {
				'Frequency' : codon_frequency(
					sequence = m_sequences[key]['CDS']['seq'],
					mrna     = key,
					relative = True
				),
				'Stability' : [0] * 8
			}

	return m_sequences, m_features

def mutate_sequence (sequence : str, mutation_rates : Dict[str, float], method : str = 'exponential') -> Tuple[str, List] :
	"""
	Doc
	"""

	RATES = {
		'mutation_rate'     : 0.05,
		'insertion_rate'    : 0.16,
		'deletion_rate'     : 0.16,
		'substitution_rate' : 0.68,
		'spread_rate'       : 0.90,
		'max_length'        : 9
	}

	RATES.update(mutation_rates)

	if RATES['mutation_rate']     is None : RATES['mutation_rate']     = 0.0
	if RATES['insertion_rate']    is None : RATES['insertion_rate']    = 0.0
	if RATES['deletion_rate']     is None : RATES['deletion_rate']     = 0.0
	if RATES['substitution_rate'] is None : RATES['substitution_rate'] = 0.0

	probabilities = Counter({
		'Insertion'    : RATES['insertion_rate']    / (RATES['insertion_rate'] + RATES['deletion_rate'] + RATES['substitution_rate']),
		'Deletion'     : RATES['deletion_rate']     / (RATES['insertion_rate'] + RATES['deletion_rate'] + RATES['substitution_rate']),
		'Substitution' : RATES['substitution_rate'] / (RATES['insertion_rate'] + RATES['deletion_rate'] + RATES['substitution_rate'])
	})

	mutation_rate = RATES['mutation_rate'] * len(sequence)
	mutation_rate = round(mutation_rate)

	template = '{:12s} @ {:' + str(len(str(len(sequence)))) + 'd} | {}'
	changes  = list()

	method = method.lower()

	if method == 'exponential' :
		mutate_method = lambda x : mutate_exponential(
			sequence = x,
			mutations = probabilities,
			template = template,
			max_length = RATES['max_length'],
			spread_rate = RATES['spread_rate']
		)
	elif method == 'random' :
		mutate_method = lambda x : mutate_random(
			sequence   = x,
			mutations  = probabilities,
			template   = template,
			max_length = RATES['max_length']
		)
	else :
		raise ValueError()

	for _ in range(mutation_rate) :
		sequence, change = mutate_method(x = sequence)
		changes.append(change)

	return sequence, changes
