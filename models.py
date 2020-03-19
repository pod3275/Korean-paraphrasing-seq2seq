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
        self.word_dim = embedtable.shape[1]
        self.emb = nn.Embedding(*embedtable.shape)
        self.emb.weight.data.copy_(embedtable)
        self.emb.weight.requires_grad = False
        self.attn = nn.Linear(HIDDEN_DIM * 2, MAX_SEQ_LEN)
        self.attn_combine = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.gru = nn.GRU(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.out = nn.Linear(HIDDEN_DIM, self.vocab_size)
        # self.hidden = nn.Parameter(self.initHidden())

        ## copy mechanism
        self.do_gen = nn.Sequential(nn.Linear(HIDDEN_DIM*2 + self.word_dim, 1), nn.Sigmoid()) # do-generation probability
        self.copy_weight_proj = nn.Linear(HIDDEN_DIM*2, 1)

        ## copy mechanism
    def get_alpha(self, e_tokens, e_states, d_states):
        # print(e_tokens.size() ,e_states.size(), d_states.size())
        mask = ~torch.eq(e_tokens, PAD_token*torch.ones(*e_tokens.size())) # bsz x e_tokens
        # get all possible cross concatenation of encoder and decoder sequence
        e_states = e_states.unsqueeze(1).repeat(1,MAX_SEQ_LEN,1,1)
        d_states = d_states.unsqueeze(2).repeat(1,1,MAX_SEQ_LEN,1)
        input_ = torch.cat([e_states, d_states], dim=3)
        # print(e_states.size(), d_states.size(), input_.size())
        logits = self.copy_weight_proj(input_).squeeze(3) # bsz x d_tokens(seq_len) x e_tokens(seq_len)
        # print(logits.size(), mask.size())
        # exit()
        # print(mask[0])
        logits += ~mask.unsqueeze(1).repeat(1,MAX_SEQ_LEN,1)*(-1e3) # add extremely small numbers to masked values (masked softmax)
        # print(logits[0])
        alphas = F.softmax(logits, dim=2)
        return alphas

    def get_context(self, e_tokens, e_states, alphas):
        context = torch.bmm(alphas, e_states) # bsz x d_tokens(seq_len) x hidden
        return context

    def forward(self, input_, encoder_inputs, encoder_hidden, encoder_outputs):
        bsz_ = input_.size(0)
        x = self.emb(input_)
        embedded = self.dropout(x)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, encoder_hidden.permute([1,0,2]).expand(-1,MAX_SEQ_LEN,-1)[:bsz_]), 2)), dim=1)
        attn_applied = torch.bmm(attn_weights,encoder_outputs) # batch x MAX_SEQ_LEN x HIDDEN_DIM
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, _ = self.gru(output, encoder_hidden)

        generation_dist = F.softmax(self.out(output), dim=2)

        alphas = self.get_alpha(encoder_inputs, encoder_outputs, output) 
        context = self.get_context(encoder_inputs, encoder_outputs, alphas)
        
        # print(alphas[0].max(1))

        # print(alphas.size(), context.size())
        # exit()

        # print(encoder_inputs.size(), alphas.size())
        # exit()
        copy_dist = torch.zeros(bsz_, MAX_SEQ_LEN, self.vocab_size).to(DEVICE).scatter_add(2, encoder_inputs.unsqueeze(2).repeat(1,1,MAX_SEQ_LEN), alphas)




        # print(copy_dist[0].max(1))
        # exit()

        mix_ratio = self.do_gen(torch.cat([output, context, embedded],dim=2))
        # print(mix_ratio.size(), generation_dist.size(), copy_dist.size())
        # exit()
        prob = generation_dist*mix_ratio +copy_dist*(1-mix_ratio)

        # print(prob.sum(2))
        # exit()

        return prob, attn_weights

