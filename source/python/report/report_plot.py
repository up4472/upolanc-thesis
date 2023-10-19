from typing import Any
from typing import Dict
from typing import Union

import seaborn
import matplotlib

from source.python.report.report_utils import convert_bert_group_to_color
from source.python.report.report_utils import convert_bert_name
from source.python.report.report_utils import convert_bert_step_to_epoch

def models_bert_r2 (data : Dict[str, Any], mode : str = 'regression', step : str = 'iteration', steps_per_epoch : Union[int, float] = 485, steps_min : int = None, steps_max : int = None, alpha : float = 0.65, linewidth : int = 2, groupby : str = None, style : str = None, filename : str = None) -> None:
	"""
	Doc
	"""

	if data is None         : return
	if data[mode] is None   : return
	if len(data[mode]) == 0 : return

	fig, ax = matplotlib.pyplot.subplots(figsize = (16, 10))
	fig.tight_layout()

	per_step  = ('step', 'Step')
	per_epoch = ('epoch', 'Epoch')

	if   step == 'iteration' : xcolumn = per_step
	elif step == 'step'      : xcolumn = per_step
	elif step == 'batch'     : xcolumn = per_step
	elif step == 'epoch'     : xcolumn = per_epoch
	else                     : xcolumn = per_step

	items = data[mode].items()
	items = sorted(items, key = lambda x : x[1]['eval_r2'].max(), reverse = True)

	for index, (name, dataframe) in enumerate(items) :
		if name.endswith('explode') :
			sitr = int(5 * steps_per_epoch)
			smin = int(5 * steps_min)
			smax = int(5 * steps_max)
		else :
			sitr = int(steps_per_epoch)
			smin = int(steps_min)
			smax = int(steps_max)

		if steps_min is not None : dataframe = dataframe[dataframe['step'] >= smin]
		if steps_max is not None : dataframe = dataframe[dataframe['step'] <= smax]

		dataframe['epoch'] = convert_bert_step_to_epoch(
			step            = dataframe['step'],
			steps_per_epoch = sitr,
			floor           = False
		)

		if   step == 'iteration' : xcolumn = ('step',  'Step')
		elif step == 'step'      : xcolumn = ('step',  'Step')
		elif step == 'epoch'     : xcolumn = ('epoch', 'Epoch')
		else                     : xcolumn = ('step',  'Step')

		name = convert_bert_name(
			name  = name,
			style = style,
			index = index
		)

		seaborn.lineplot(
			data      = dataframe,
			x         = xcolumn[0],
			y         = 'eval_r2',
			ax        = ax,
			label     = name,
			alpha     = alpha,
			linewidth = linewidth,
			color = convert_bert_group_to_color(
				name    = name,
				groupby = groupby
			)
		)

	ax.set_ylabel('R2')
	ax.set_xlabel(xcolumn[1])

	if filename is not None :
		matplotlib.pyplot.savefig(
			filename + '.png',
			format      = 'png',
			dpi         = 120,
			bbox_inches = 'tight',
			pad_inches  = 0
		)
