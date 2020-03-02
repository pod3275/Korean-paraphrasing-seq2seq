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
from eval import evaluateRandomly
from utils import prepareData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_name = "paraphrasing data_DH.xlsx"

dictionary, pair_data = prepareData("kor", file_name)
embedtable = np.loadtxt("word_emb.txt", delimiter=" ", dtype='float32')
embedtable = np.append(embedtable, np.random.rand(1, 128).astype('float32'), axis=0) # '<EQ>'


encoder = Encoder(dictionary.n_tokens, 128, embedtable).to(device)
attndecoder = AttnDecoder(128, dictionary.n_tokens, embedtable, dropout_p=0.1).to(device)

trainIters(encoder, attndecoder, dictionary, pair_data, epochs=100)

evaluateRandomly(encoder, attndecoder, pair_data, dictionary, n=10)
