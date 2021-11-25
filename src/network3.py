# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel

# +
### featurizer 모음
# -

class AudioFeaturizer3(torch.nn.Module):   # LSTM
    ### mfcc 벡터를 LSTM에 통과 시켜서 last hidden state return
    def __init__(self):
        super(AudioFeaturizer3, self).__init__()
        
        self.L2 = 256
        self.cell_units = 128
        self.num_linear = 20
        self.p = 10
        
        self.linear1 = nn.Linear(self.p*self.L2, self.num_linear) # [10*256, 768]
        self.bn = nn.BatchNorm1d(self.num_linear)
        self.lstm = nn.LSTM(input_size=self.num_linear, hidden_size=2048, num_layers=2, batch_first=True)

        ### attention layers
        self.a_fc1 = nn.Linear(2*self.cell_units, 1)  
        self.a_fc2 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, l):   # 여기서 x는 list of tensor list(tensor)
        
        #x = pad_sequence(x, batch_first=True, padding_value=0)
        x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        # x, state = self.lstm(x)     # state: hidden state

        ### lstm
        print(x.shape)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, self.time_step, self.L2*self.p)        # (-1, 150, 256*10)
        x = x.reshape(-1, self.L2*self.p)                        # (1500, 2560)

        x = self.relu(self.bn(self.linear1(x)))                 # [1500, 768]
        x = x.reshape(-1, self.time_step, self.num_linear)     # [10, 150, 768]

        x, state = self.lstm(x)                       # outputs1 : [10, 150, 128] (B,T,D)

        ### attention
        v = self.sigmoid(self.a_fc1(state[0][1]))                  # (10, 150, 1)
        print(state[0][1].shape)
        print(v.shape)
        
        alphas = self.softmax(self.a_fc2(v).squeeze())   # (B,T) shape, alphas are attention weights
        
        print(alphas)
        x = (alphas.unsqueeze(2) * state[0][1]).sum(axis=1)      # (B,D)

        return x
