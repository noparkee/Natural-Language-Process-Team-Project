# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.models
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# +
### featurizer 모음
# -

class AudioFeaturizer3(torch.nn.Module):   # LSTM
    ### mfcc 벡터를 LSTM에 통과 시켜서 last hidden state return
    def __init__(self):
        super(AudioFeaturizer3, self).__init__()

        self.input_size = 20
        self.hidden_size = 2048
        self.output_size = 768
        self.max_length = 1121


        self.encoder = EncoderRNN(input_size=self.input_size, hidden_size=self.hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size=self.hidden_size, output_size=self.output_size, dropout_p=0.1, max_length=self.max_length)

    def forward(self, x, l):   # 여기서 x는 list of tensor list(tensor)
        
        #x = pad_sequence(x, batch_first=True, padding_value=0)
        x = pack_padded_sequence(x, max(l), batch_first=True, enforce_sorted=False)
        
        self.hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(max(x), self.hidden_size, device=device)
        x = self.encoder(x, self.hidden)
        x = self.decoder(x, self.hidden, encoder_outputs)

        return x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length = 1121):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

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


