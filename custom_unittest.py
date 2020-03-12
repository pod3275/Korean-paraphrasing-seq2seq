import numpy as np
from torch.optim import Adam
import torch.nn as nn

# custom modules
from utils import *
from config import *
from train import *
from models import *



def test_loader():
	file_name = "paraphrasing data_DH.xlsx"
	dictionary, pair_data = prepareData("kor", file_name)
	ld = Loader(dictionary, pair_data)

	for batch in ld.train:
		assert batch[0].size(0) == BATCH_SIZE 
		assert batch[0].size(1) == MAX_SEQ_LEN 
		assert batch[1].size(0) == BATCH_SIZE 
		assert batch[1].size(1) == MAX_SEQ_LEN 
		break
	

	for batch in ld.test:
		assert batch[0].size(0) == BATCH_SIZE 
		assert batch[0].size(1) == MAX_SEQ_LEN 
		assert batch[1].size(0) == BATCH_SIZE 
		assert batch[1].size(1) == MAX_SEQ_LEN 
		break

	print('*******batch loader test passed!*******')

def test_train_epoch():
	VOCAB_SIZE = 35000
	file_name = "paraphrasing data_DH.xlsx"
	dictionary, pair_data = prepareData("kor", file_name)
	ld = Loader(dictionary, pair_data)

	embed = torch.from_numpy(np.random.rand(VOCAB_SIZE,WORD_DIM)).float()
	enc = Encoder(embed)
	dec = AttnDecoder(embed)

	encoder_optim = Adam(enc.parameters(), lr = LEARNING_RATE)
	decoder_optim = Adam(dec.parameters(), lr = LEARNING_RATE)
	criterion = nn.NLLLoss()

	for e in range(NUM_EPOCH):
		loss = train_epoch(ld.train, enc, dec, encoder_optim, decoder_optim, criterion)
		print('loss : {}'.format(loss))
		break

	print('*******train_epoch test passed!*******')


def test_encoder():
	VOCAB_SIZE = 35000
	batch = [torch.randint(1,VOCAB_SIZE,size=(BATCH_SIZE, MAX_SEQ_LEN))
			]

	embed = torch.from_numpy(np.random.rand(VOCAB_SIZE,WORD_DIM)).float()
	enc = Encoder(embed)

	out, hidden = enc(*batch)

	assert out.size(0) == BATCH_SIZE
	assert out.size(1) == MAX_SEQ_LEN
	assert out.size(2) == HIDDEN_DIM

	print('*******encoder test passed!*******')

def test_decoder():
	VOCAB_SIZE = 35000
	batch = [torch.randint(1,VOCAB_SIZE,size=(BATCH_SIZE, MAX_SEQ_LEN)),
			torch.zeros(NUM_LAYER*(int(BIDIRECTIONAL)+1), BATCH_SIZE, HIDDEN_DIM),
			torch.randn(BATCH_SIZE, MAX_SEQ_LEN,HIDDEN_DIM)]

	embed = torch.from_numpy(np.random.rand(VOCAB_SIZE,WORD_DIM)).float()
	dec = AttnDecoder(embed)

	out, hidden = dec(*batch)

	assert out.size(0) == BATCH_SIZE
	assert out.size(1) == MAX_SEQ_LEN
	assert out.size(2) == VOCAB_SIZE

	print('*******decoder test passed!*******')


def test_indexes_from_sentence():
	test_sentences = ['여기서 젤라틴의 수소 이온 지수(pH)를 (미지수)라 할 때, 젤라틴의 두께를 (미지수)를 사용한 식으로 나타내면 (수식)이다.',
						'직교 : 두 직선 (미지수)와 (미지수)의 교각이 직각일 때, 두 직선은 서로 직교한다고 한다. (화살표) 기호 (수식)',
						'또, (수식) (등호) (수식) (등호) (수식) 이면 (수식)이 성립해요.']


	file_name = "paraphrasing data_DH.xlsx"
	dic_, pair_data = prepareData("kor", file_name)

	for tc in test_sentences:
		print(indexesFromSentence(dic_, tc))

if __name__ == '__main__':
	test_loader()
	test_encoder()
	test_decoder()
	test_train_epoch()
	test_indexes_from_sentence()