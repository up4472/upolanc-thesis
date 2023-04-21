from pandas import DataFrame
from typing import Dict
from typing import List

import numpy

def get_statistics_for (dataframe : DataFrame, transcripts : List[str], group : str, data : Dict[str, Dict] = None, axis : int = 1) -> Dict[str, Dict] :
	"""
	Doc
	"""

	matrix  = dataframe.iloc[:, 1:].values

	if group is None : group = 'Global'
	else             : group = group.title()

	if data is None :
		data = {
			'Mean'     : dict(),
			'Min'      : dict(),
			'Max'      : dict(),
			'Median'   : dict(),
			'StDev'    : dict(),
			'Variance' : dict(),
			'P75'      : dict(),
			'P25'      : dict(),
			'Range'    : dict()
		}

	mat_mean     = numpy.mean(matrix, axis = axis)
	mat_median   = numpy.median(matrix, axis = axis)
	mat_stdev    = numpy.std(matrix, axis = axis)
	mat_variance = numpy.var(matrix, axis = axis)
	mat_min      = numpy.min(matrix, axis = axis)
	mat_max      = numpy.max(matrix, axis = axis)
	mat_p25      = numpy.percentile(matrix, 25, axis = axis)
	mat_p75      = numpy.percentile(matrix, 75, axis = axis)
	mat_range    = numpy.ptp(matrix, axis = axis)

	for index, transcript in enumerate(transcripts) :
		data['Mean'    ][(transcript, group)] = mat_mean[index]
		data['Median'  ][(transcript, group)] = mat_median[index]
		data['StDev'   ][(transcript, group)] = mat_stdev[index]
		data['Variance'][(transcript, group)] = mat_variance[index]
		data['Min'     ][(transcript, group)] = mat_min[index]
		data['Max'     ][(transcript, group)] = mat_max[index]
		data['P25'     ][(transcript, group)] = mat_p25[index]
		data['P75'     ][(transcript, group)] = mat_p75[index]
		data['Range'   ][(transcript, group)] = mat_range[index]

	return data

def get_statistics_dataframe (data : Dict[str, Dict]) -> DataFrame :
	"""
	Doc
	"""

	dataframe = DataFrame.from_dict(data)
	dataframe.index = dataframe.index.set_names(['Transcript', 'Tissue'])

	dataframe = dataframe.reset_index()

	dataframe.set_index(['Transcript', 'Tissue'], inplace = True)
	dataframe.sort_index(inplace = True)

	return dataframe
