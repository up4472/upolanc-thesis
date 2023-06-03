TARGETS = [
	'global-mean',
	'tissue-mean', 'tissue-mean-explode', 'tissue-mean-seedling',
	 'group-mean',  'group-mean-explode',  'group-mean-young_seedling'
]

CNNS         = ['zrimec', 'washburn']
FILTERS      = ['f1', 'f2', 'f3', 'f4', 'f5']
SEQUENCES    = ['bp512', 'bp2150', 'bp6150']
OPTIMIZERS   = ['adam', 'lamb']
FEATURES     = [0, 64, 72, 77]
EPOCHS       = [25, 50, 100, 150, 200, 250, 500, 1000, 2000]
TRIALS       = [500, 1000, 2000]
KMERS        = [3, 6]
BERT_ARCH    = ['def', 'rnn', 'cat']
BERT_OUTPUT  = ['fc2', 'fc3']
BERT_LAYERS  = [6, 9, 11, 12]
FLOAT_FORMAT = '{:.5f}'
