from Bio.Data import IUPACData

from collections import Counter
from typing      import List
from typing      import Tuple

import random

def insertion (sequence : str, index : int, curr : List[str], prev : List[str], letters : str) -> Tuple[str, List[str], List[str], int] :
	"""
	Doc
	"""

	head = sequence[:index]
	item = random.choice(letters)
	tail = sequence[index:]

	mutation = head + item + tail

	curr.append(mutation[index])

	return mutation, curr, prev, index + 1

def deletion (sequence : str, index : int, curr : List[str], prev : List[str], letters : str) -> Tuple[str, List[str], List[str], int] : # noqa
	"""
	Doc
	"""

	head = sequence[:index]
	tail = sequence[index + 1:]

	mutation = head + tail

	prev.append(sequence[index])

	return mutation, curr, prev, index

def substitution (sequence : str, index : int, curr : List[str], prev : List[str], letters : str) -> Tuple[str, List[str], List[str], int] :
	"""
	Doc
	"""

	head = sequence[:index]
	item = random.choice(letters)
	tail = sequence[index + 1:]

	mutation = head + item + tail

	prev.append(sequence[index])
	curr.append(mutation[index])

	return mutation, curr, prev, index

def mutate (sequence : str, mutations : Counter, template : str, spread_rate : float = None, spread_limit : int = None) -> Tuple[str, str] :
	"""
	Doc
	"""

	if spread_limit is None :
		spread_limit = len(sequence)

	nucleotides = IUPACData.unambiguous_dna_letters

	mtype = random.choices(*zip(*mutations.items()), k = 1)[0]
	start = random.randint(0, len(sequence) - 1)
	index = start

	prev = list()
	curr = list()

	spread_count = 0

	while index < len(sequence) :
		match mtype :
			case 'Insertion'    : function = insertion
			case 'Deletion'     : function = deletion
			case 'Substitution' : function = substitution
			case _ : raise ValueError()

		sequence, curr, prev, index = function(
			sequence = sequence,
			index    = index,
			curr     = curr,
			prev     = prev,
			letters  = nucleotides
		)

		spread_count = spread_count + 1

		if spread_rate is None :
			break

		if spread_rate <= random.random() :
			break

		if spread_limit <= spread_count :
			break

	match mtype :
		case 'Insertion'    : mutation = ''.join(curr)
		case 'Deletion'     : mutation = ''.join(prev)
		case 'Substitution' : mutation = ''.join(prev) + ' -> ' + ''.join(curr)
		case _ : raise ValueError()

	return sequence, template.format(mtype, start, mutation)
