from torch                    import Tensor
from torch.nn                 import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from types                    import FunctionType
from typing                   import Any
from typing                   import Dict

from tqdm.auto import tqdm

import numpy
import torch

from src.cnn.callbacks.savebest import SaveBestModel
from src.cnn.callbacks.savelast import SaveLastModel

def compute_metrics (metrics : Dict[str, FunctionType], outputs : Tensor, labels : Tensor, report : Dict[str, Dict]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	if 'metric' not in report.keys() :
		report['metric'] = dict()

	for key, metric in metrics.items() :
		data = metric(outputs, labels)
		data = data.detach().cpu().numpy()

		if key not in report['metric'].keys() :
			report['metric'][key] = data
		else :
			stack = report['metric'][key]

			if stack is None :
				report['metric'][key] = data
			else :
				report['metric'][key] = numpy.vstack((stack, data))

	return report

def train_epoch (model : Module, params : Dict[str, Any], desc : str = 'Progress') -> Dict[str, Any] :
	"""
	Doc
	"""

	model.train(mode = True)

	dataloader = params['train_dataloader']
	optimizer = params['optimizer']
	criterion = params['criterion']
	metrics = params['metrics']
	device = params['device']
	verbose = params['verbose']

	batch_loss = 0.0
	batch_ids = list()
	batch_report = {
		'metric' : dict(),
		'loss' : numpy.nan
	}

	for metric in metrics.keys() :
		batch_report['metric'][metric] = None

	progbar = tqdm(dataloader, disable = not verbose)
	progbar.set_description_str(desc = desc)

	for batch_index, batch in enumerate(progbar, start = 1) :
		ids, inputs, features, labels = batch

		inputs   = inputs.to(device)
		features = features.to(device)
		labels   = labels.to(device)

		optimizer.zero_grad()

		outputs = model(inputs, features)
		loss = criterion(outputs, labels)

		batch_report = compute_metrics(
			metrics = metrics,
			outputs = outputs,
			labels  = labels,
			report  = batch_report
		)

		loss.backward()
		optimizer.step()

		batch_ids.extend(ids)
		batch_loss = batch_loss + loss.item()
		print_loss = batch_loss / batch_index

		progbar.set_description_str(
			desc = f'{desc} | Loss = {print_loss: 8.5f}'
		)

	batch_report['batch'] = numpy.array(batch_ids)
	batch_report['loss'] = batch_loss / len(dataloader)

	return batch_report

def evaluate_epoch (model : Module, params : Dict[str, Any], desc : str = 'Progress', validation : bool = False) -> Dict[str, Any] :
	"""
	Doc
	"""

	model.train(mode = False)
	model.eval()

	if validation :
		dataloader = params['valid_dataloader']
	else :
		dataloader = params['test_dataloader']

	criterion = params['criterion']
	metrics = params['metrics']
	device = params['device']
	verbose = params['verbose']

	batch_loss = 0.0
	batch_genes = list()
	batch_ypred = list()
	batch_ytrue = list()

	batch_report = {
		'metric' : dict(),
		'loss' : numpy.nan,
		'genes' : list(),
		'ypred' : list(),
		'ytrue' : list()
	}

	for metric in metrics.keys() :
		batch_report['metric'][metric] = None

	progbar = tqdm(dataloader, disable = not verbose)
	progbar.set_description_str(desc = desc)

	with torch.no_grad() :
		for batch_index, batch in enumerate(progbar, start = 1) :
			ids, inputs, features, labels = batch

			inputs   = inputs.to(device)
			features = features.to(device)
			labels   = labels.to(device)

			outputs = model(inputs, features)
			loss = criterion(outputs, labels)

			batch_report = compute_metrics(
				metrics = metrics,
				outputs = outputs,
				labels  = labels,
				report  = batch_report
			)

			if not validation :
				batch_genes.extend(ids)
				batch_ypred.extend(outputs.detach().cpu().numpy())
				batch_ytrue.extend(labels.detach().cpu().numpy())

			batch_loss = batch_loss + loss.item()
			print_loss = batch_loss / batch_index

			progbar.set_description_str(
				desc = f'{desc} | Loss = {print_loss: 8.5f}'
			)

		batch_report['genes'] = numpy.array(batch_genes)
		batch_report['ypred'] = numpy.array(batch_ypred)
		batch_report['ytrue'] = numpy.array(batch_ytrue)
		batch_report['loss'] = batch_loss / len(dataloader)

	return batch_report

def train (model : Module, params : Dict[str, Any]) -> Dict[str, Dict | numpy.ndarray] :
	"""
	Doc
	"""

	scheduler = params['scheduler']
	optimizer = params['optimizer']
	criterion = params['criterion']
	metrics = params['metrics']
	device = params['device']
	epochs = params['epochs']

	model = model.to(device)

	savebest = None if params['savebest'] is None else SaveBestModel(filename = params['savebest'])
	savelast = None if params['savelast'] is None else SaveLastModel(filename = params['savelast'])

	report = dict()

	for mode in ['train', 'valid'] :
		report[mode] = dict()
		report[mode]['metric'] = dict()
		report[mode]['loss'] = list()
		report[mode]['lr'] = list()

		for metric in metrics.keys() :
			report[mode]['metric'][metric] = list()

	description = str(len(str(epochs)))
	description = 'Epoch {:0' + description + 'd}/{:0' + description + 'd}'

	for epoch in range(epochs) :
		ogdesc = description.format(1 + epoch, epochs)
		lrcurr = optimizer.param_groups[0]['lr']
		lrdesc = 'LR = {:.2e}'.format(lrcurr)

		train_report = train_epoch(
			model  = model,
			params = params,
			desc   = ogdesc + ' | Train | ' + lrdesc
		)

		valid_report = evaluate_epoch(
			model      = model,
			params     = params,
			validation = True,
			desc       = ogdesc + ' | Valid | ' + lrdesc
		)

		train_loss = train_report['loss']
		valid_loss = valid_report['loss']

		if savebest is not None :
			savebest.update(
				model     = model,
				optimizer = optimizer,
				criterion = criterion,
				epoch     = 1 + epoch,
				loss      = valid_loss
			)

		if scheduler is not None :
			if isinstance(scheduler, ReduceLROnPlateau) :
				scheduler.step(valid_loss)
			else :
				scheduler.step()

		for metric in metrics.keys() :
			report['train']['metric'][metric].append(train_report['metric'][metric])
			report['valid']['metric'][metric].append(valid_report['metric'][metric])

		report['train']['loss'].append(train_loss)
		report['valid']['loss'].append(valid_loss)
		report['train']['lr'].append(lrcurr)
		report['valid']['lr'].append(lrcurr)

	if savelast is not None :
		savelast.update(
			model     = model,
			optimizer = optimizer,
			criterion = criterion,
			epoch     = epochs,
			loss      = report['valid']['loss'][-1]
		)

	report['train']['loss'] = numpy.array(report['train']['loss'])
	report['valid']['loss'] = numpy.array(report['valid']['loss'])
	report['train']['lr'] = numpy.array(report['train']['lr'])
	report['valid']['lr'] = numpy.array(report['valid']['lr'])

	return report

def evaluate (model : Module, params : Dict[str, Any]) -> Dict[str, Dict] :
	"""
	Doc
	"""

	return {
		'eval' : evaluate_epoch(
			model      = model,
			params     = params,
			desc       = 'Evaluation',
			validation = False
		)
	}
