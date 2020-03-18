import torch
import numpy as np
from torch.optim import Adam
import sys

# custom modules
from models import *
from train import *
from eval import *
from utils import *


file_name = "new_data.xlsx"

dictionary, pair_data = prepareData("kor", file_name)
ld = Loader(dictionary, pair_data)
embedtable = np.loadtxt("word_emb.txt", delimiter=" ", dtype='float32')
special_embeddings = np.concatenate((np.random.rand(len(SPECIAL_TOKENS)-1, 128).astype('float32'),
	np.zeros((1,128), dtype=np.float32)), axis=0)
embedtable = np.insert(embedtable, [2], special_embeddings, axis=0) 
embedtable = torch.from_numpy(embedtable).float()


enc = Encoder(embedtable).to(DEVICE)
dec = AttnDecoder(embedtable).to(DEVICE)

encoder_optim = Adam(enc.parameters(), lr = LEARNING_RATE)
decoder_optim = Adam(dec.parameters(), lr = LEARNING_RATE)
criterion = nn.NLLLoss(reduction='none')

for e in range(NUM_EPOCH):
	loss = train_epoch(ld.train, enc, dec, encoder_optim, decoder_optim, criterion)
	if (e+1) % (SAMPLE_TEST_EVERY//10) == 0 : 
		print('EPOCH {} - loss : {}\r'.format(e+1,loss))
	if (e+1) % SAMPLE_TEST_EVERY == 0: 
		sample_train_sentences, sample_test_sentences = sample_sentences(pair_data)
		tr_infer = infer_sentence(enc, dec, [item[0] for item in sample_train_sentences], dictionary) # train infer
		ts_infer = infer_sentence(enc, dec, [item[0] for item in sample_test_sentences], dictionary) # test infer
		print('='*10,'Train set inference', '='*10)
		print_result(tr_infer, sample_train_sentences)
		print('='*10,'Test set inference', '='*10)
		print_result(ts_infer, sample_test_sentences)

