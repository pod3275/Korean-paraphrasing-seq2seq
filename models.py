# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:38:35 2020

@author: 이상헌
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 100

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedtable):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedtable = torch.from_numpy(embedtable)
        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
#        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
        embedded = self.embedtable[input, :].view(1, -1) # view=reshape
        output = torch.unsqueeze(embedded, 0)
        if device.type == 'cuda':
            output = output.cuda()
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
#class Decoder(nn.Module):
#    def __init__(self, hidden_size, output_size):
#        super(Decoder, self).__init__()
#        self.hidden_size = hidden_size
#        
#        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
#        self.embedding = nn.Embedding(output_size, hidden_size)
#        self.lstm = nn.LSTM(hidden_size, hidden_size)
#        self.out = nn.Linear(hidden_size, output_size)
#        self.softmax = nn.LogSoftmax(dim=1)
#        
#    def forward(self, input, hidden):
#        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
#        output = self.embedding(input).view(1, 1, -1)
#        output = F.relu(output)
#        output, hidden = self.lstm(output, hidden)
#        output = self.softmax(self.out(output[0]))
#        
#    def initHidden(self):
#        return torch.zeros(1, 1, self.hidden_size, device=device)
        
    
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedtable, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedtable = torch.from_numpy(embedtable)

        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
#        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
        embedded = self.embedtable[input, :].view(1, -1)
        embedded = torch.unsqueeze(embedded, 0)
        embedded = self.dropout(embedded)

        if device.type == 'cuda':
            embedded = embedded.cuda()
            
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)