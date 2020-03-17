import numpy as np
import copy
DELETE_TOKEN = ["@ToBeRemoved"]

def calcNgram(sentence, n):
    # Returns n gram array
    # np.shape(sentence) = (n,)
    n_gram_list = []
    for i in range(len(sentence) - n + 1):
        n_gram_list.append(sentence[i:i + n])
    return n_gram_list


def calcPrecision(ref_n_gram_list, cand_n_gram_list):
    tot = np.shape(cand_n_gram_list)[0]
    recall = 0.0
    for n_words in ref_n_gram_list:
        if n_words in cand_n_gram_list:
            recall += 1
    return recall/tot


def doesInclude(listA, listB):
    #longer B is recommended
    if len(listA)==0:
        return 0
    for a in listA:
        if a not in listB:
            return 0
    return 1


def sentenceRepetitionRemover(sentence):
    sentence_tmp = sentence.copy()
    for word in sentence_tmp:
        counter=[]
        for i in range(len(sentence)):
            if word == sentence[i]:
                counter.append(i)
        if len(counter)>1:
            for c in range(len(counter)):
                sentence_tmp[counter[c]] = str(sentence_tmp[counter[c]])+'@'+str(c)
    return sentence_tmp




def calcChunks(ref_sentence, cand_sentence):
    ref_sentence = sentenceRepetitionRemover(ref_sentence)
    cand_sentence = sentenceRepetitionRemover(cand_sentence)
    chunks_tmp = []
    cand_ngrams = [calcNgram(cand_sentence, n + 1) for n in range(len(cand_sentence))]
    ref_ngrams = [calcNgram(ref_sentence, n + 1) for n in range(len(ref_sentence))]
    for i in range(len(cand_ngrams)):
        cand_ngram_list = cand_ngrams[i]
        chunk_tmp = []
        for cand_ngram in cand_ngram_list:
            for j in range(len(ref_ngrams)):
                if cand_ngram in ref_ngrams[j]:
                    chunk_tmp.append(cand_ngram)
        if len(chunk_tmp):
            chunks_tmp.append(chunk_tmp)
    chunks = copy.deepcopy(chunks_tmp)
    for i in range(len(chunks_tmp)-1):
        for j in range(len(chunks_tmp[i])-1):
            for k in range(len(chunks_tmp[i+1])):
                if(doesInclude(chunks_tmp[i][j]+chunks_tmp[i][j+1], chunks_tmp[i+1][k])):
                    chunks[i][j] = DELETE_TOKEN
                    chunks[i][j + 1] = DELETE_TOKEN
                    continue

    chunks_result =[]

    for i in range(len(chunks)):
        for j in range(len(chunks[i])):
            if not DELETE_TOKEN[0] in chunks[i][j]:
                chunks_result.append(chunks[i][j])
    return chunks_result


class Metrics(object):
    """def __init__(self, reference_sentences, candidate_sentences):
        self.reference_sentences = reference_sentences
        self.candidate_sentences = candidate_sentences"""
    def calcRouge(self, ref_sentence, cand_sentence, n):
        ref_n_gram_list = calcNgram(ref_sentence, n)
        cand_n_gram_list = calcNgram(cand_sentence, n)
        tot = np.shape(ref_n_gram_list)[0]
        recall=0.0
        for n_words in ref_n_gram_list:
            if n_words in cand_n_gram_list:
                recall+=1
        return recall/tot

    def calcBleu(self, ref_sentence, cand_sentence):

        #####################Calculate precision#######################
        ref_ngram1 = calcNgram(ref_sentence, 1)
        cand_ngram1 = calcNgram(cand_sentence, 1)
        self.precision1 = calcPrecision(ref_ngram1, cand_ngram1)

        ref_ngram2 = calcNgram(ref_sentence, 2)
        cand_ngram2 = calcNgram(cand_sentence, 2)
        self.precision2 = calcPrecision(ref_ngram2, cand_ngram2)

        ref_ngram3 = calcNgram(ref_sentence, 3)
        cand_ngram3 = calcNgram(cand_sentence, 3)
        self.precision3 = calcPrecision(ref_ngram3, cand_ngram3)

        ref_ngram4 = calcNgram(ref_sentence, 4)
        cand_ngram4 = calcNgram(cand_sentence, 4)
        self.precision4 = calcPrecision(ref_ngram4, cand_ngram4)

        self.precision =(self.precision1*self.precision2*self.precision3*self.precision4)**(1.0/4.0)
        #####################Calculate precision#######################

        self.brev_penalty = np.min([1.0, len(cand_sentence)/len(ref_sentence)])
        return self.brev_penalty*self.precision

    def calcMeteor(self, ref_sentence, cand_sentence, r=0.1):
        if r<0 or r>1:
            print("r must be a number between 0 and 1")
            return -1
        ref_ngram1 = calcNgram(ref_sentence, 1)
        cand_ngram1 = calcNgram(cand_sentence, 1)
        m = 0.0
        for word in cand_ngram1:
            if word in ref_ngram1:
                m += 1
        w_t = len(cand_ngram1)
        w_r = len(ref_ngram1)

        P = m/w_t
        R = m/w_r
        self.F_mean = P*R/(r*R + (1-r)*P)

        self.u_m = 0.0
        for word in ref_ngram1:
            if word in cand_ngram1:
                self.u_m += 1
        self.c = len(calcChunks(ref_sentence, cand_sentence))

        p = 0.5*(self.c/self.u_m)**3
        self.score = self.F_mean*(1-p)

        return self.score

    def printScores(self, ref_sentence, cand_sentence, r=0.1):
        if len(ref_sentence)<4 or len(cand_sentence)<4:
            print('sentence length should be longer than 4')
            return -1
        print("ROUGE-1: {}".format(self.calcRouge(ref_sentence, cand_sentence, 1)))
        print("ROUGE-2: {}".format(self.calcRouge(ref_sentence, cand_sentence, 2)))
        print("ROUGE-3: {}".format(self.calcRouge(ref_sentence, cand_sentence, 3)))
        print("ROUGE-4: {}".format(self.calcRouge(ref_sentence, cand_sentence, 4)))

        print("BLEU: {}".format(self.calcBleu(ref_sentence, cand_sentence)))

        print("METEOR: {}".format(self.calcMeteor(ref_sentence, cand_sentence, r)))

        return
