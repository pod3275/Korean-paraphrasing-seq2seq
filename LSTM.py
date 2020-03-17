# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:36:26 2020

@author: 이상헌
"""

from __future__ import unicode_literals, print_function, division

import torch
import numpy as np

from models import Encoder, AttnDecoder
from train import trainIters
from eval import evaluateRandomly, evaluate
from utils import *
from config import *

file_name = "new_data.xlsx"

dictionary, pair_data = prepareData("kor", file_name, artif_token=True)
embedtable = np.loadtxt("word_emb.txt", delimiter=" ", dtype='float32')
special_embeddings = np.concatenate((np.random.rand(len(SPECIAL_TOKENS)-1, 128).astype('float32'),
	np.zeros((1,128), dtype=np.float32)), axis=0)
embedtable = np.insert(embedtable, [2], special_embeddings, axis=0)
embedtable = torch.from_numpy(embedtable).float()

encoder = Encoder(dictionary.n_tokens, 128, embedtable).to(DEVICE)
attndecoder = AttnDecoder(128, dictionary.n_tokens, embedtable, dropout_p=DROPOUT_RATIO).to(DEVICE)

trainIters(encoder, attndecoder, dictionary, 
           pair_data, epochs=NUM_EPOCH, batch_size=BATCH_SIZE,
           print_examples = PRINT_EXAMPLES)

evaluateRandomly(encoder, attndecoder, pair_data, dictionary, n=PRINT_EXAMPLES)

sentence = "(문어체) 수량 사이의 관계를 파악하여 이차방정식으로 나타낸다."
print('>', sentence)
decoded_words, _ = evaluate(encoder, attndecoder, sentence, dictionary)