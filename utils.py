# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:38:35 2020

@author: 이상헌
"""

from __future__ import unicode_literals, print_function, division

import time
import math
import random
import pandas as pd
import torch

from gluonnlp.data import SentencepieceTokenizer
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from kobert.utils import get_tokenizer

from utils import *
from config import *


SPECIAL_TOKENS = ['<EXPR>','<UNVAR>','<EQUL>','<ARRW>','<WTN>','<CLQ>','<PAD>']
KOREAN_2_SPECIAL = {'(수식)'  :'\N{Arabic Poetic Verse Sign}',
                    '(미지수)':'\N{Arabic Sign Misra}' ,
                    '(화살표)':'\N{Arabic Place of Sajdah}',
                    '(등호)'  :'\N{Left-Facing Armenian Eternity Sign}',
                    '(문어체)':'\N{Arabic Start of Rub El Hizb}',
                    '(구어체)':'\N{Right-Facing Armenian Eternity Sign}'}
SPECIAL_2_ENG = dict(zip(['\N{Arabic Poetic Verse Sign}',
                          '\N{Arabic Sign Misra}' ,
                          '\N{Arabic Place of Sajdah}',
                          '\N{Left-Facing Armenian Eternity Sign}',
                          '\N{Arabic Start of Rub El Hizb}',
                          '\N{Right-Facing Armenian Eternity Sign}'], SPECIAL_TOKENS[:6]))

            
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.index2token = {} 
        self.n_tokens = 0
#        tok_path = get_tokenizer()
        tok_path = "C:/KoBERT/kobert_news_wiki_ko_cased-1087f8699e.spiece"
        self.sp = SentencepieceTokenizer(tok_path)
                      
    def addSentence(self, sentence):
        for token in self.sp(sentence):
            self.addToken(token)
        
    def generateIdx(self):
        with open("word_list.txt", "r", encoding="UTF8") as f:
            self.index2token = [tok.strip() for tok in f]
        self.index2token[2:2] = SPECIAL_TOKENS
        self.token2index = {v : k for k,v in enumerate(self.index2token)}
        self.n_tokens = len(self.index2token)
                
            
def indexesFromSentence(dictionary, sentence):
    for k,v in KOREAN_2_SPECIAL.items(): # replace special tokens
        sentence = sentence.replace(k,v)
    tokens = [token for token in dictionary.sp(sentence)]
    tokens = [SPECIAL_2_ENG[ele] if ele in SPECIAL_2_ENG else ele for ele in tokens]
    return [dictionary.token2index[token] for token in tokens]


def tensorFromSentence(dictionary, sentence):
    indexes = indexesFromSentence(dictionary, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(pair, dictionary):
    input_tensor = tensorFromSentence(dictionary, pair[0])
    target_tensor = tensorFromSentence(dictionary, pair[1])
    return (input_tensor, target_tensor)


def readLangs(name, file_name, artif_token=True):
    print("Reading lines...")

    # 파일을 읽고 줄로 분리
    file = pd.read_excel("Data/" + file_name)
    
    if artif_token:
        pairs = [['(문어체) '+file['train_x'][i], '(구어체) '+file['train_y'][i]] for i in range(len(file))]
        pairs_r = [list(reversed(p)) for p in pairs]
        pairs = pairs + pairs_r
    
    else:
        pairs = [[file['train_x'][i], file['train_y'][i]] for i in range(len(file))]

    # 쌍을 뒤집고, Lang 인스턴스 생성
    dictionary = Dictionary(name)
    dictionary.generateIdx()

    return dictionary, pairs


def prepareData(name, file_name, artif_token=True):
    dictionary, pairs = readLangs(name, file_name, artif_token)
    random.shuffle(pairs)
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
    
