import torch
import numpy as np
from torch.optim import Adam
import sys

# custom modules
from models import *
from train import *
from eval import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_name = "paraphrasing data_DH.xlsx"

dictionary, pair_data = prepareData("kor", file_name)
ld = Loader(dictionary, pair_data)
embedtable = np.loadtxt("word_emb.txt", delimiter=" ", dtype='float32')
special_embeddings = np.random.rand(len(SPECIAL_TOKENS), 128).astype('float32')
embedtable = np.append(embedtable, special_embeddings, axis=0) 
embedtable = torch.from_numpy(embedtable).float()


enc = Encoder(embedtable).to(device)
dec = AttnDecoder(embedtable).to(device)

encoder_optim = Adam(enc.parameters(), lr = LEARNING_RATE)
decoder_optim = Adam(dec.parameters(), lr = LEARNING_RATE)
criterion = nn.NLLLoss()

for e in range(NUM_EPOCH):
	loss = train_epoch(ld.train, enc, dec, encoder_optim, decoder_optim, criterion)
	sys.stdout.write('EPOCH {} - loss : {}\r'.format(e+1,loss))
	# if (e+1) % == 0: 
		# evaluateRandomly()

