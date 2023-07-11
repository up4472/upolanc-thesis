from distinctipy import distinctipy

TARGETS = [
	'global-mean',
	'tissue-mean', 'tissue-mean-explode', 'tissue-mean-seedling',
	 'group-mean', 'group-mean-explode',  'group-mean-young_seedling'
]

COLORS         = distinctipy.get_colors(n_colors = 10, pastel_factor = 0.4)
COLORS         = [(int(255 * r), int(255 * g), int(255 * b)) for r, g, b in COLORS]
COLORS         = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in COLORS]
CNNS           = ['zrimec', 'washburn']
FILTERS        = ['f1', 'f2', 'f3', 'f4', 'f5']
SEQUENCES      = ['tf2150', 'tf6150', 'po0512', 'po4096', 'pu4096']
OPTIMIZERS     = ['adam', 'lamb']
FEATURES       = [0, 64, 72, 77]
EPOCHS         = [25, 50, 100, 150, 200, 250, 500, 1000, 2000]
TRIALS         = [250, 500, 750, 1000, 2000]
KMERS          = [3, 4, 5, 6]
BERT_ARCH      = ['def', 'rnn', 'cat', 'fex']
BERT_POOLER    = ['v1', 'v2']
BERT_OUTPUT    = ['fc2', 'fc3']
BERT_LAYERS    = [6, 9, 11, 12]
FLOAT_FORMAT   = '{:.5f}'

# tf2150 | Transcript Full 2150 bp with 1000 bp promoter
# tf6150 | Transcript Full 6150 bp with 5000 bp promoter
# po0512 |   Promoter Only  512 bp
# po4096 |   Promoter Only 4096 bp
# pu4096 |   Promoter UTR5 4096 bp

# def - DNABert with 1 x 512 size
# rnn - DNABert with n x 512 size; uses RNN
# cat - DNABert with n x 512 size; uses Concatenate
# fex - DNABert feature extractor
