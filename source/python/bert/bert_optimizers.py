from tensorboardX import SummaryWriter
from torch.optim  import Optimizer

import collections
import torch

def log_lamb_rs (optimizer : Optimizer, event_writer : SummaryWriter, token_count : int) :
	"""
	Doc
	"""

	results = collections.defaultdict(list)

	for group in optimizer.param_groups :
		for p in group['params'] :
			state = optimizer.state[p]

			for i in ('weight_norm', 'adam_norm', 'trust_ratio') :
				if i in state :
					results[i].append(state[i])

	for k, v in results.items() :
		event_writer.add_histogram(f'lamb/{k}', torch.tensor(v), token_count)

class Lamb (Optimizer) :

	def __init__ (self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-6, weight_decay = 0, adam = False) :
		"""
		Doc
		"""

		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

		defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)

		self.adam = adam

		super(Lamb, self).__init__(params, defaults)

	def step (self, closure = None) :
		"""
		Doc
		"""

		loss = None

		if closure is not None :
			loss = closure()

		for group in self.param_groups :
			for p in group['params'] :
				if p.grad is None :
					continue

				grad = p.grad.data

				if grad.is_sparse:
					raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

				state = self.state[p]

				if len(state) == 0:
					state['step']       = 0
					state['exp_avg']    = torch.zeros_like(p.data)
					state['exp_avg_sq'] = torch.zeros_like(p.data)

				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				beta1, beta2 = group['betas']

				state['step'] += 1

				exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

				step_size = group['lr']

				weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
				adam_step   = exp_avg / exp_avg_sq.sqrt().add(group['eps'])

				if group['weight_decay'] != 0:
					adam_step.add_(p.data, alpha=group['weight_decay'])

				adam_norm = adam_step.pow(2).sum().sqrt()

				if weight_norm == 0 or adam_norm == 0 : trust_ratio = 1
				else                                  : trust_ratio = weight_norm / adam_norm

				state['weight_norm'] = weight_norm
				state['adam_norm'] = adam_norm
				state['trust_ratio'] = trust_ratio

				if self.adam:
					trust_ratio = 1

				p.data.add_(adam_step, alpha=-step_size * trust_ratio)

		return loss
