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

def mutate_random (sequence : str, mutations : Counter, template : str, max_length : int) -> Tuple[str, str] :
	"""
	Doc
	"""

	nucleotides = IUPACData.unambiguous_dna_letters

	mtype = random.choices(*zip(*mutations.items()), k = 1)[0]
	start = random.randint(0, len(sequence) - 1)
	index = start

	prev = list()
	curr = list()

	max_length = random.randint(1, max_length)

	for _ in range(max_length) :
		if   mtype == 'Insertion'    : function = insertion
		elif mtype == 'Deletion'     : function = deletion
		elif mtype == 'Substitution' : function = substitution
		else : raise ValueError()

		sequence, curr, prev, index = function(
			sequence = sequence,
			index    = index,
			curr     = curr,
			prev     = prev,
			letters  = nucleotides
		)

		if index >= len(sequence) :
			break

	if   mtype == 'Insertion'    : mutation = ''.join(curr)
	elif mtype == 'Deletion'     : mutation = ''.join(prev)
	elif mtype == 'Substitution' : mutation = ''.join(prev) + ' -> ' + ''.join(curr)
	else : raise ValueError()

	return sequence, template.format(mtype, start, mutation)

def mutate_exponential (sequence : str, mutations : Counter, template : str, max_length : int, spread_rate : float) -> Tuple[str, str] :
	"""
	Doc
	"""

	nucleotides = IUPACData.unambiguous_dna_letters

	mtype = random.choices(*zip(*mutations.items()), k = 1)[0]
	start = random.randint(0, len(sequence) - 1)
	index = start

	prev = list()
	curr = list()

	for _ in range(max_length) :
		if   mtype == 'Insertion'    : function = insertion
		elif mtype == 'Deletion'     : function = deletion
		elif mtype == 'Substitution' : function = substitution
		else : raise ValueError()

		sequence, curr, prev, index = function(
			sequence = sequence,
			index    = index,
			curr     = curr,
			prev     = prev,
			letters  = nucleotides
		)

		if index >= len(sequence) :
			break

		if spread_rate <= random.random() :
			break

	if   mtype == 'Insertion'    : mutation = ''.join(curr)
	elif mtype == 'Deletion'     : mutation = ''.join(prev)
	elif mtype == 'Substitution' : mutation = ''.join(prev) + ' -> ' + ''.join(curr)
	else : raise ValueError()

	return sequence, template.format(mtype, start, mutation)
