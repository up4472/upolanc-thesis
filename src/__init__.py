import numpy
import pandas
import seaborn

numpy.set_printoptions(
	suppress  = True,
	edgeitems = 25,
	linewidth = 150,
	formatter = {
		'float_kind' : '{: 7,.3f}'.format
	}
)

pandas.set_option(
	'display.float_format',
	'{:.3f}'.format
)

seaborn.set_theme()
