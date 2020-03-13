from __future__ import unicode_literals, print_function, division
import random

from utils import tensorFromSentence
import torch

#custom modules
from utils import *
from config import *

    
def infer_sentence(encoder, decoder, sentences, dictionary):
    # sentences : list of sentences to infer

    with torch.no_grad():
        input_tensor = [tensorFromSentence(dictionary, sent) for sent in sentences]
        
        # input sentence padding / trimming
        for i, ts in enumerate(input_tensor):
            if ts.size(0) >= MAX_SEQ_LEN : input_tensor[i] = ts[:MAX_SEQ_LEN]
            else: input_tensor[i] = torch.cat((ts, torch.empty(MAX_SEQ_LEN - ts.size(0), dtype=torch.long).fill_(PAD_token)), 0)

        input_tensor = torch.stack(input_tensor, dim = 0)

        
        encoder_output, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([[SOS_token]+[PAD_token for _ in range(MAX_SEQ_LEN-1)] for _ in range(input_tensor.size(0))])
        
        for pos in range(MAX_SEQ_LEN-1):
            decoder_output, decoder_attention = decoder(decoder_input, encoder_hidden, encoder_output)
            decoder_input[:,pos+1] = decoder_output.max(2)[1][:,pos]

        ret = []
        for d_o in decoder_input:
            decoded_sent = ''.join([dictionary.index2token[item] for item in d_o.tolist()])
            decoded_sent = decoded_sent.replace('▁',' ')
            decoded_sent = decoded_sent.split('<eos>')[0] + ' <eos>'
            ret.append(decoded_sent)

        return ret

