import torch

BATCH_SIZE= 32
TEST_RATIO = 0.15
SEED = 20200311
MAX_SEQ_LEN = 100
USE_TEACHER_FORCING = True
SOS_token = 0
EOS_token = 1
PAD_token = 6
BIDIRECTIONAL = False
HIDDEN_DIM = 128
DROPOUT = 0.2
WORD_DIM = 128
NUM_LAYER = 1
LEARNING_RATE= 0.01
NUM_EPOCH = 1000
NUM_SAMPLE = 3


SPECIAL_TOKENS = ['<EXPR>','<UNVAR>','<EQUL>', '<ARRW>','<PAD>']
KOREAN_2_SPECIAL = {'(수식)':'\N{Arabic Poetic Verse Sign}',
					 '(미지수)':'\N{Arabic Sign Misra}' ,
					 '(등호)':'\N{Arabic Place of Sajdah}',
					 '(화살표)':'\N{Arabic Sign Sindhi Ampersand}'}
SPECIAL_2_ENG = dict(zip(['\N{Arabic Poetic Verse Sign}',
					 '\N{Arabic Sign Misra}' ,
					 '\N{Arabic Place of Sajdah}',
					 '\N{Arabic Sign Sindhi Ampersand}'], SPECIAL_TOKENS[:4]))

SAMPLE_TEST_EVERY = 1
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
