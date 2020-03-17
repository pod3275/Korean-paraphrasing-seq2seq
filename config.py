# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:40:15 2020

@author: 이상헌
"""

import torch

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 100

DROPOUT_RATIO = 0.1
NUM_EPOCH = 30
BATCH_SIZE = 64
TEACHER_FORCING_RATIO = 0.5

PRINT_EXAMPLES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp = open("results.txt", "w", encoding='utf-8')