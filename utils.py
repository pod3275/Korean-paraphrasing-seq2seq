import time
import math
import pandas as pd
import torch
import random
from torch.utils import data
import numpy as np

from gluonnlp.data import SentencepieceTokenizer
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from kobert.utils import get_tokenizer


# custom modules
from config import *

    
            
class Dictionary:
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.index2token = {} 
        self.n_tokens = 0
        tok_path = get_tokenizer()
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

class Loader:
    def __init__(self, dictionary, pairs):
        # PAD_token = dictionary.n_tokens + (SPECIAL_TOKENS.index('<PAD>') - len(SPECIAL_TOKENS))
        self.pad = lambda x: torch.cat([x, torch.empty(MAX_SEQ_LEN - x.size(0), dtype=torch.long).fill_(PAD_token)], dim = 0)
        self.trim = lambda x: x[:MAX_SEQ_LEN].contiguous()
        self.dic_ = dictionary 
        self.train, self.test = self.get_train_test_loader(pairs) 

    def sent_into_numeric(self, sentence):
        ten_ = tensorFromSentence(self.dic_,sentence)
        return self.trim(ten_) if ten_.size(0) > MAX_SEQ_LEN else self.pad(ten_) 

    def get_train_test_loader(self, pairs_):
        # shuffle pairs
        random.seed(SEED)
        pairs = [item for item in pairs_] # copy
        random.shuffle(pairs)

        # set idx
        test_pos = int(len(pairs)*TEST_RATIO)

        # make loader
        return data.DataLoader([tuple(map(self.sent_into_numeric,item)) for item in pairs[:-test_pos]],batch_size=BATCH_SIZE), data.DataLoader([tuple(map(self.sent_into_numeric,item)) for item in pairs[-test_pos:]],batch_size=BATCH_SIZE)

def sample_sentences(pairs_):
    random.seed(SEED)
    pairs = [item for item in pairs_] # copy
    random.shuffle(pairs)

        # set idx
    test_pos = int(len(pairs)*TEST_RATIO)
    tr_sents = np.array(pairs)[np.random.randint(0,test_pos, NUM_SAMPLE)]
    ts_sents = np.array(pairs)[np.random.randint(test_pos, len(pairs), NUM_SAMPLE)]

    return tr_sents.tolist(), ts_sents.tolist()


def indexesFromSentence(dictionary, sentence):
    for k,v in KOREAN_2_SPECIAL.items(): # replace special tokens
        sentence = sentence.replace(k,v)
    tokens = [token for token in dictionary.sp(sentence)]
    tokens = [SPECIAL_2_ENG[ele] if ele in SPECIAL_2_ENG else ele for ele in tokens]
    # print(tokens)
    return [dictionary.token2index[token] for token in tokens]


def tensorFromSentence(dictionary, sentence):
    indexes = indexesFromSentence(dictionary, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)


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
    
def print_result(inference_list, x_y_pair_list):
    for y_hat, x_y_pair in zip(inference_list, x_y_pair_list):
        print('x : {}'.format(x_y_pair[0]),
            'y_hat : {}'.format(y_hat),
            'y : {}'.format(x_y_pair[1]),'-'*30, sep='\n')

