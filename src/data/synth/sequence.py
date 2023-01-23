from collections import Counter
from typing      import Dict
from typing      import List
from typing      import Tuple

from src.data.synth._mutation import mutate

def mutate_sequence (sequence : str, mutation_rates : Dict[str, float]) -> Tuple[str, List] :
	"""
	Doc
	"""

	RATES = {
		'mutation_rate'     : 0.05,
		'insertion_rate'    : 1,
		'deletion_rate'     : 1,
		'substitution_rate' : 4,
		'spread_rate'       : None,
		'spread_limit'      : None
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

	for _ in range(mutation_rate) :
		sequence, change = mutate(
			sequence     = sequence,
			mutations    = probabilities,
			template     = template,
			spread_rate  = RATES['spread_rate'],
			spread_limit = RATES['spread_limit']
		)

		changes.append(change)

	return sequence, changes
