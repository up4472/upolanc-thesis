from transformers import AlbertConfig                          # noqa F821 :: unresolved reference :: added at runtime
from transformers import AlbertForSequenceClassification       # noqa F821 :: unresolved reference :: added at runtime
from transformers import AlbertTokenizer                       # noqa F821 :: unresolved reference :: added at runtime
from transformers import BertConfig                            # noqa F821 :: unresolved reference :: added at runtime
from transformers import BertForSequenceClassification         # noqa F821 :: unresolved reference :: added at runtime
from transformers import BertForLongSequenceClassification     # noqa F821 :: unresolved reference :: added at runtime
from transformers import BertTokenizer                         # noqa F821 :: unresolved reference :: added at runtime
from transformers import DNATokenizer                          # noqa F821 :: unresolved reference :: added at runtime
from transformers import DistilBertConfig                      # noqa F821 :: unresolved reference :: added at runtime
from transformers import DistilBertForSequenceClassification   # noqa F821 :: unresolved reference :: added at runtime
from transformers import DistilBertTokenizer                   # noqa F821 :: unresolved reference :: added at runtime
from transformers import FlaubertConfig                        # noqa F821 :: unresolved reference :: added at runtime
from transformers import FlaubertForSequenceClassification     # noqa F821 :: unresolved reference :: added at runtime
from transformers import FlaubertTokenizer                     # noqa F821 :: unresolved reference :: added at runtime
from transformers import RobertaConfig                         # noqa F821 :: unresolved reference :: added at runtime
from transformers import RobertaForSequenceClassification      # noqa F821 :: unresolved reference :: added at runtime
from transformers import RobertaTokenizer                      # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLMConfig                             # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLMForSequenceClassification          # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLMRobertaConfig                      # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLMRobertaForSequenceClassification   # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLMRobertaTokenizer                   # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLMTokenizer                          # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLNetConfig                           # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLNetForSequenceClassification        # noqa F821 :: unresolved reference :: added at runtime
from transformers import XLNetTokenizer                        # noqa F821 :: unresolved reference :: added at runtime

from transformers import glue_output_modes                     # noqa F821 :: unresolved reference :: added at runtime
from transformers import glue_processors                       # noqa F821 :: unresolved reference :: added at runtime

from source.python.bert.bert_models     import FeatureExtractorBert
from source.python.bert.bert_models     import RegressionBertFC1
from source.python.bert.bert_models     import RegressionBertFC3
from source.python.bert.bert_models     import CatRegressionBertFC3
from source.python.bert.bert_models     import RnnRegressionBertFC3
from source.python.bert.bert_processors import RegressionProcessor

PRETRAINED_MODELS = sum((
	tuple(conf.pretrained_config_archive_map.keys())
	for conf in (
		      BertConfig,  XLNetConfig,        XLMConfig,  RobertaConfig,
		DistilBertConfig, AlbertConfig, XLMRobertaConfig, FlaubertConfig
	)),
	()
)

MODELS = {
	# Original
	'dna'          : (      BertConfig,       BertForSequenceClassification,        DNATokenizer),
	'dnalong'      : (      BertConfig,   BertForLongSequenceClassification,        DNATokenizer),
	'dnalongcat'   : (      BertConfig,   BertForLongSequenceClassification,        DNATokenizer),
	'bert'         : (      BertConfig,       BertForSequenceClassification,       BertTokenizer),
	'xlnet'        : (     XLNetConfig,      XLNetForSequenceClassification,      XLNetTokenizer),
	'xlm'          : (       XLMConfig,        XLMForSequenceClassification,        XLMTokenizer),
	'roberta'      : (   RobertaConfig,    RobertaForSequenceClassification,    RobertaTokenizer),
	'distilbert'   : (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
	'albert'       : (    AlbertConfig,     AlbertForSequenceClassification,     AlbertTokenizer),
	'xlmroberta'   : (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
	'flaubert'     : (  FlaubertConfig,   FlaubertForSequenceClassification,   FlaubertTokenizer),
	# Custom
	'febert'       : (      BertConfig,             FeatureExtractorBert,           DNATokenizer),
	'rbertfc1'     : (      BertConfig,                   RegressionBertFC1,        DNATokenizer),
	'rbertfc3'     : (      BertConfig,                   RegressionBertFC3,        DNATokenizer),
	'rbertfc3_def' : (      BertConfig,                   RegressionBertFC3,        DNATokenizer),
	'rbertfc3_cat' : (      BertConfig,                CatRegressionBertFC3,        DNATokenizer),
	'rbertfc3_rnn' : (      BertConfig,                RnnRegressionBertFC3,        DNATokenizer)
}

TOKENS = [
	'bert',
	'dnalong',
	'dnalongcat',
	'xlnet',
	'albert'
]

PROCESSORS = glue_processors   | {'regression' : RegressionProcessor}
MODES      = glue_output_modes | {'regression' : 'regression'}
