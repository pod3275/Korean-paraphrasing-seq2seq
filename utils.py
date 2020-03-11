# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:38:35 2020

@author: 이상헌
"""

from __future__ import unicode_literals, print_function, division

import time
import math
import pandas as pd
import torch
import random
from torch.utils import data

from gluonnlp.data import SentencepieceTokenizer
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from kobert.utils import get_tokenizer


# config
BATCH_SIZE = 32 
TEST_RATIO = 0.15
SEED = 20200311
MAX_SEQ_LEN = 50

SOS_token = 0
EOS_token = 1
PAD_token = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok_path = get_tokenizer()
sp  = SentencepieceTokenizer(tok_path)
    
            
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.index2token = {} 
        self.n_tokens = 0
                      
    def addSentence(self, sentence):
        for token in sp(sentence):
            self.addToken(token)
        
    def generateIdx(self):
        with open("word_list.txt", "r", encoding="UTF8") as f:
            for num, token in enumerate(f):
                token = token.strip()
                self.index2token[num] = token
                self.token2index[token] = num
                self.n_tokens+=1
            
        self.index2token[self.n_tokens] = '<EQ>'
        self.token2index['<EQ>'] = self.n_tokens
        self.n_tokens+=1
                

class Loader:
    def __init__(self, dictionary, pairs):
        self.pad = lambda x: torch.cat([x, torch.empty(MAX_SEQ_LEN - x.size(0), dtype=torch.long).fill_(PAD_token)], dim = 0)
        self.trim = lambda x: x[:MAX_SEQ_LEN].contiguous()
        self.dic_ = dictionary 
        self.train, self.test = self.get_train_test_loader(pairs) 

    def sent_into_numeric(self, sentence):
        ten_ = tensorFromSentence(self.dic_,sentence)
        return self.trim(ten_) if ten_.size(0) > MAX_SEQ_LEN else self.pad(ten_) 

    def get_train_test_loader(self, pairs):
        # shuffle pairs
        random.seed(SEED)
        random.shuffle(pairs)

        # set idx
        test_pos = int(len(pairs)*TEST_RATIO)

        # make loader
        return data.DataLoader([tuple(map(self.sent_into_numeric,item)) for item in pairs[:-test_pos]],batch_size=BATCH_SIZE), data.DataLoader([tuple(map(self.sent_into_numeric,item)) for item in pairs[-test_pos:]],batch_size=BATCH_SIZE)
            
def indexesFromSentence(dictionary, sentence):
    tokens = [token for token in sp(sentence)]
    new_tokens = []
    i=0
    while i<len(tokens):
        if i < len(tokens)-3 and (tokens[i] == '▁(' or tokens[i] == '(') and tokens[i+1] == '수' and tokens[i+2] == '식' and tokens[i+3] == ')':
            new_tokens.append('<EQ>')
            i= i+4
        else:
            new_tokens.append(tokens[i])
            i+=1
    
    return [dictionary.token2index[token] for token in new_tokens]


def tensorFromSentence(dictionary, sentence):
    indexes = indexesFromSentence(dictionary, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensorsFromPair(pair, dictionary):
    input_tensor = tensorFromSentence(dictionary, pair[0])
    target_tensor = tensorFromSentence(dictionary, pair[1])
    return (input_tensor, target_tensor)


def readLangs(name, file_name, reverse=False):
    print("Reading lines...")

    # 파일을 읽고 줄로 분리
    file = pd.read_excel("Data/" + file_name)
    pairs = [[file['train_x'][i], file['train_y'][i]] for i in range(len(file))]

    # 쌍을 뒤집고, Lang 인스턴스 생성
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        dictionary = Dictionary(name)
        dictionary.generateIdx()
    else:
        dictionary = Dictionary(name)
        dictionary.generateIdx()

    return dictionary, pairs


def prepareData(name, file_name, reverse=False):
    dictionary, pairs = readLangs(name, file_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counted words:")
    print(dictionary.name, dictionary.n_tokens)
    return dictionary, pairs


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 주기적인 간격에 이 locator가 tick을 설정
    loc = ticker.MultipleLocator(base=1)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
