from pandas import DataFrame

import numpy

def clean_annotation (dataframe : DataFrame) -> DataFrame :
	"""
	Doc
	"""

	dataframe = dataframe.drop(columns = [
		'Alias',
		'biotype',
		'constitutive',
		'description',
		'ensembl_end_phase',
		'ensembl_phase',
		'ID',
		'Is_circular',
		'logic_name',
		'Name',
		'protein_id',
		'Phase',
		'rank',
		'Score',
		'Source'
	])

	dataframe = dataframe.rename(columns = {
		'Seqid'         : 'Seq',
		'exon_id'       : 'Exon',
		'gene_id'       : 'Gene',
		'transcript_id' : 'Transcript'
	})

	dataframe = dataframe.astype({
		'Start' : int,
		'End'   : int
	})

	dataframe['Type'] = dataframe['Type'].str.replace('five_prime_UTR', 'UTR5')
	dataframe['Type'] = dataframe['Type'].str.replace('three_prime_UTR', 'UTR3')
	dataframe['Type'] = dataframe['Type'].str.replace('exon', 'Exon')
	dataframe['Type'] = dataframe['Type'].str.replace('gene', 'Gene')

	dataframe = dataframe.loc[dataframe['Type'].isin(['mRNA', 'UTR3', 'CDS', 'UTR5'])].copy()

	dataframe['Parent'] = dataframe['Parent'].str.split(':').str[-1]
	dataframe['Parent'] = dataframe['Parent'].str.upper()

	dataframe['Transcript'] = dataframe['Transcript'].fillna(dataframe['Parent'])
	dataframe['Transcript'] = dataframe['Transcript'].str.split(':').str[-1]
	dataframe['Transcript'] = dataframe['Transcript'].str.upper()

	dataframe['Gene'] = dataframe['Gene'].fillna(dataframe['Transcript'])
	dataframe['Gene'] = dataframe['Gene'].str.split('.').str[0]
	dataframe['Gene'] = dataframe['Gene'].str.upper()

	dataframe['Length'] = numpy.absolute(dataframe['End'] - dataframe['Start'])

	return dataframe[['Seq', 'Strand', 'Type', 'Gene', 'Transcript', 'Exon', 'Parent', 'Start', 'End', 'Length']]

	# dataframe['Length'] = abs(dataframe['End'] - dataframe['Start']) + 1
	#
	# dataframe['Type'] = dataframe['Type'].str.replace('five_prime_UTR', 'UTR5')
	# dataframe['Type'] = dataframe['Type'].str.replace('three_prime_UTR', 'UTR3')
	# dataframe['Type'] = dataframe['Type'].str.replace('exon', 'Exon')
	# dataframe['Type'] = dataframe['Type'].str.replace('gene', 'Gene')
	#
	# dataframe = dataframe.loc[dataframe['Type'].isin(['mRNA', 'UTR3', 'CDS', 'UTR5'])].copy()
	#
	# dataframe['Parent'] = dataframe['Parent'].apply(lambda x : x.split(':')[1] if isinstance(x, str) else x)
	# dataframe['mRNA'] = dataframe['mRNA'].apply(lambda x : x.split(':')[1] if isinstance(x, str) else x)
	#
	# dataframe['mRNA'].fillna(dataframe['Parent'], inplace = True)
	# dataframe['Gene'].fillna(dataframe['Parent'], inplace = True)
	#
	# dataframe['Gene'] = dataframe['Gene'].apply(lambda x : x.split('.')[0] if isinstance(x, str) else x)
	#
	# dataframe.loc[dataframe['Type'] == 'Gene', 'mRNA'] = numpy.nan
	#
	# dataframe['mRNA'] = dataframe['mRNA'].apply(lambda x : x.upper() if isinstance(x, str) else x)
	# dataframe['Gene'] = dataframe['Gene'].apply(lambda x : x.upper() if isinstance(x, str) else x)
	#
	# return dataframe[['Seq', 'Strand', 'Type', 'Gene', 'mRNA', 'Start', 'End', 'Length']]

def clean_metadata (dataframe : DataFrame) -> DataFrame :
	"""
	Doc
	"""

	dataframe = dataframe.drop(columns = [
		'age',
		'BioProject',
		'BioSample',
		'checked',
		'comments',
		'complete_time',
		'experimental_design',
		'GEO_series',
		'Library_name',
		'keep',
		'perturbation_experiment',
		'perturbation_group_1',
		'perturbation_group_1_1',
		'perturbation_group_2_1',
		'perturbation_group_2_2',
		'perturbation_group_2_3',
		'priority',
		'project_description',
		'project_title',
		'SRX_accession',
		'Sample',
		'QC_summary',
		'species_strain',
		'summary_Bioporoject_QC',
		'time_course',
		'time_series',
		'tissue',
		'Unnamed: 32'
	])

	dataframe = dataframe.rename(columns = {
		'age_group'          : 'Age',
		'control'            : 'Control',
		'perturbation_group' : 'Perturbation',
		'senescence samples' : 'Senescence',
		'SRAStudy'           : 'Study',
		'SRR_accession'      : 'Sample',
		'tissue_group'       : 'Tissue',
		'tissue_super'       : 'Group'
	})

	dataframe = dataframe.loc[~dataframe['Tissue'].isin(['drop'])].copy()

	dataframe['Tissue'] = dataframe['Tissue'].str.replace('other tissue', 'other')
	dataframe['Group']  = dataframe['Group'].str.replace('senescence_senescence_green', 'senescence_green')
	dataframe['Group']  = dataframe['Group'].str.replace('senescence_senescence_reproductive', 'senescence_reproductive')
	dataframe['Group']  = dataframe['Group'].str.replace('mature_other tissue', 'mature_other')
	dataframe['Group']  = dataframe['Group'].str.replace('young_other tissue', 'young_other')

	dataframe['Senescence'] = dataframe['Senescence'].fillna(value = 'no')
	dataframe['Perturbation'] = dataframe['Perturbation'].str.split().str[0]
	#dataframe['Perturbation'] = dataframe['Perturbation'].apply(lambda x : x.split()[0] if isinstance(x, str) else x)

	return dataframe[['Sample', 'Study', 'Control', 'Senescence', 'Age', 'Tissue', 'Group', 'Perturbation']]

def clean_tpm (dataframe : DataFrame) -> DataFrame :
	"""
	Doc
	"""

	dataframe = dataframe.rename(columns = {
		'gene_id' : 'Transcript'
	})

	dataframe.columns = [
		x.split('_')[0]
		for x in dataframe.columns.tolist()
	]

	return dataframe
