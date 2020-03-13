import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class Encoder(nn.Module):
    def __init__(self, embedtable):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(*embedtable.shape)
        self.emb.weight.data.copy_(embedtable)
        self.emb.weight.requires_grad = False
        self.gru = nn.GRU(WORD_DIM, HIDDEN_DIM, batch_first=True)
        self.hidden = nn.Parameter(self.initHidden())
        
    def forward(self, input_):
        bsz_ = input_.size(0)
        x = self.emb(input_)
        output, hidden = self.gru(x, self.hidden[:,:bsz_])
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(NUM_LAYER*(int(BIDIRECTIONAL)+1), BATCH_SIZE, HIDDEN_DIM).to(DEVICE)
            
    
class AttnDecoder(nn.Module):
    def __init__(self, embedtable):
        super(AttnDecoder, self).__init__()
        self.vocab_size = embedtable.shape[0]
        self.emb = nn.Embedding(*embedtable.shape)
        self.emb.weight.data.copy_(embedtable)
        self.emb.weight.requires_grad = False
        self.attn = nn.Linear(HIDDEN_DIM * 2, MAX_SEQ_LEN)
        self.attn_combine = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.gru = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.out = nn.Linear(HIDDEN_DIM, self.vocab_size)
        # self.hidden = nn.Parameter(self.initHidden())

    def forward(self, input_, encoder_hidden, encoder_outputs):
        # embedding함수 --> lookup table에서 찾는 걸로 바꾸기
        bsz_ = input_.size(0)
        x = self.emb(input_)
        embedded = self.dropout(x)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, encoder_hidden.permute([1,0,2]).expand(-1,MAX_SEQ_LEN,-1)[:bsz_]), 2)), dim=1)
        attn_applied = torch.bmm(attn_weights,encoder_outputs) # batch x MAX_SEQ_LEN x HIDDEN_DIM
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, _ = self.gru(output, encoder_hidden)

        output = F.log_softmax(self.out(output), dim=2)
        return output, attn_weights
