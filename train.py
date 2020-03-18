# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:38:35 2020

@author: 이상헌
"""

from __future__ import unicode_literals, print_function, division
import random, time
import torch
import torch.nn as nn
from torch import optim
from utils import tensorsFromPair, timeSince, showPlot
from math import ceil
from eval import evaluateRandomly

teacher_forcing_ratio = 0.5

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(input_tensors, target_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    target_length_total=0

    for d in range(len(input_tensors)):
        input_tensor = input_tensors[d]
        target_tensor = target_tensors[d]

        encoder_hidden = encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]]).to(device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing 포함: 목표를 다음 입력으로 전달
            # input, hidden, e_tokens, encoder_outputs
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, input_tensor,encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, input_tensor ,encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # 입력으로 사용할 부분을 히스토리에서 분리

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
        
        target_length_total+=target_length

    (loss/len(input_tensors)).backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length_total


def trainIters(encoder, decoder, dictionary, pairs, epochs, print_every=1000, print_sentences=5, learning_rate=0.1, batch_size=16):#?
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # print_every 마다 초기화
    plot_loss_total = 0  # plot_every 마다 초기화

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(pairs[i], dictionary)
                      for i in range(len(pairs))]
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        num_batch = ceil(len(pairs) // batch_size)
        print_loss_total = 0

        for b in range(num_batch):
            if b==num_batch-1:
                num_data = len(pairs)-batch_size*b
            else:
                num_data = batch_size
                
            input_tensors = [training_pairs[m][0].to(device) for m in range(batch_size*b, batch_size*b+num_data)]
            target_tensors = [training_pairs[m][1].to(device) for m in range(batch_size*b, batch_size*b+num_data)]
            
            loss = train(input_tensors, target_tensors, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss

            print('%s epochs, %s step, %.4f' % (e, batch_size*b+num_data, loss))
            
        print_loss_avg = print_loss_total / num_batch
        print('%s epochs, %.4f' % (e, print_loss_avg))
        
        evaluateRandomly(encoder, decoder, pairs, dictionary, n=print_sentences)
        
