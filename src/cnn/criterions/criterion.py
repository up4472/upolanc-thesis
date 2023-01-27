from torch    import Tensor
from torch.nn import Module
from typing   import Callable

from src.cnn.criterions import R2Score

class WeightedCriterion (Module) :

	def __init__ (self, criterion : Callable, reduction : str = 'mean', weights : Tensor = None, **kwargs) -> None :
		"""
		Doc
		"""

		super(WeightedCriterion, self).__init__()

		self.vectorized = False
		self.reduction  = reduction.lower()
		self.weights    = weights

		if self.reduction not in ['none', 'mean', 'sum'] :
			raise ValueError()

		if weights is not None :
			self.criterion = criterion(reduction = 'none', **kwargs)
		else :
			self.criterion = criterion(reduction = self.reduction, **kwargs)

		if isinstance(self.criterion, R2Score) :
			self.vectorized = True

	def forward (self, inputs : Tensor, labels : Tensor) -> Tensor :
		"""
		Doc
		"""

		score = self.criterion(inputs, labels)

		if self.weights is not None and self.reduction != 'none' :
			if not self.vectorized :
				score = torch.mean(score, dim = 0)

			score = torch.dot(self.weights, score)

		return score

if __name__ == '__main__' :
	import torch
	import numpy

	x = [[0.28, 0.34, 0.31], [0.24, 0.35, 0.30]]
	y = [[0.25, 0.34, 0.30], [0.24, 0.39, 0.29]]
	w1 = [1/3, 1/3, 1/3] # uniform
	w2 = [0.1, 0.8, 0.1] # weighted

	x  = torch.tensor(numpy.array(x))
	y  = torch.tensor(numpy.array(y))
	w1 = torch.tensor(numpy.array(w1))
	w2 = torch.tensor(numpy.array(w2))

	# c = R2Score
	c = torch.nn.L1Loss

	ERROR_EPS = 1e-12

	# reduction = none
	gt = c(reduction = 'none')(x, y)

	none1 = WeightedCriterion(criterion = c, reduction = 'none', weights = None)
	none2 = WeightedCriterion(criterion = c, reduction = 'none', weights = w2)

	score1 = none1(x, y)
	score2 = none2(x, y)

	print()
	print(gt.detach().cpu().numpy())
	print(score1.detach().cpu().numpy())
	print(score2.detach().cpu().numpy())
	print()

	if torch.sum(abs(gt - score1)) < ERROR_EPS :
		print('Criterion(handle, none, None)    passed test')
	if torch.sum(abs(gt - score2)) < ERROR_EPS :
		print('Criterion(handle, none, weight)  passed test')

	print()
	print('!------')

	# reduction = mean
	gt = c(reduction = 'mean')(x, y)

	mean1 = WeightedCriterion(criterion = c, reduction = 'mean', weights = None)
	mean2 = WeightedCriterion(criterion = c, reduction = 'mean', weights = w1)

	score1 = mean1(x, y)
	score2 = mean2(x, y)

	print()
	print(gt.detach().cpu().numpy())
	print(score1.detach().cpu().numpy())
	print(score2.detach().cpu().numpy())
	print()

	if abs(gt - score1) < ERROR_EPS :
		print('Criterion(handle, mean, None)    passed test')
	if abs(gt - score2) < ERROR_EPS :
		print('Criterion(handle, mean, uniform) passed test')

	print()
	print('!------')

	# reduction = mean + weighted
	score1 = WeightedCriterion(criterion = c, reduction = 'mean', weights = w2)(x, y)

	print()
	print(score1.detach().cpu().numpy())
