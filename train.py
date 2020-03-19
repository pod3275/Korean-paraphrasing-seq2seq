from __future__ import unicode_literals, print_function, division
import random, time
import torch
import torch.nn as nn
from torch import optim
from utils import tensorsFromPair, timeSince, showPlot
from math import ceil

# custom module
from config import *



def train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    VOCAB_SIZE = encoder.emb.weight.size(0)
    loss = 0.
    num_data = 0.

    for batch in train_loader:
        bsz_ = batch[0].size(0) # get batch size
        batch = [ele.to(DEVICE) for ele in batch] # mount tensors to device

        encoder_output, encoder_hidden = encoder(batch[0])
        decoder_input = torch.cat((torch.tensor([[SOS_token] for _ in range(bsz_)]).to(DEVICE),
            batch[1][:,:MAX_SEQ_LEN-1])
            , dim=1)
        # Teacher forcing 포함: 목표를 다음 입력으로 전달
        decoder_output, decoder_attention = decoder(
            decoder_input, encoder_hidden, encoder_output)

        mask = ~torch.eq(batch[1], PAD_token*torch.ones(batch[1].size()))
        loss_tmp = criterion(decoder_output.view(-1,VOCAB_SIZE), batch[1].view(-1))
        loss_tmp = loss_tmp*mask.view(-1)
        loss += loss_tmp.sum()/mask.sum()

        num_data += batch[0].size(0)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/num_data
        