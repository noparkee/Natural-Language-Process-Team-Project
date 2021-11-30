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

# +
### featurizer 모음
# -

class AudioFeaturizer(torch.nn.Module):   # LSTM
    ### mfcc 벡터를 LSTM에 통과 시켜서 last hidden state return
    def __init__(self):
        super(AudioFeaturizer, self).__init__()
        
        blstm_hidden = 1024
        self.blstm = nn.LSTM(input_size=20, hidden_size=blstm_hidden, num_layers=2, batch_first=True, bidirectional=True)     # 양방향

        self.cnn_0 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2)
        self.batchNorm_1 = nn.BatchNorm1d(num_features=128)

        self.cnn_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.cnn_3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.maxPool_4 = nn.MaxPool1d(kernel_size=2)
        self.batchNorm_5 = nn.BatchNorm1d(num_features=128)

        # self.cnn_6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.cnn_7 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.maxPool_8 = nn.MaxPool1d(kernel_size=2)
        # self.batchNorm_9 = nn.BatchNorm1d(num_features=512)

        # self.cnn_10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.cnn_11 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        # self.maxPool_12 = nn.MaxPool1d(kernel_size=2)
        # self.batchNorm_13 = nn.BatchNorm1d(num_features=1024)

        self.cnn_j = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1)
        self.cnn_k = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1)
        self.cnn_l = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)

        self.gamma = nn.parameter.Parameter(torch.zeros(1))

    def forward(self, x, l):
        #x = pad_sequence(x, batch_first=True, padding_value=0)
        x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        x, state = self.blstm(x)
        x, _ = pad_packed_sequence(x, total_length=max(l), batch_first=True)

        y = torch.cat((state[0][-2], state[0][-1]), dim=1)    # (B, 2 * h)

        y = y.unsqueeze(dim=1)  # (B, 1, 1024)
        y = F.relu(self.cnn_0(y))  # (B, 128, 1024)
        
        y = self.batchNorm_1(y)

        y = F.relu(self.cnn_2(y))
        y = F.relu(self.cnn_3(y))  # (B, 256, 1024)
        y = self.maxPool_4(y)   # (B, 256, 512)
        y = self.batchNorm_5(y)

        # y = F.relu(self.cnn_6(y))
        # y = F.relu(self.cnn_7(y))  # (B, 512, 512)
        # y = self.maxPool_8(y)   # (B, 512, 256)
        # y = self.batchNorm_9(y)

        # y = F.relu(self.cnn_10(y))
        # y = F.relu(self.cnn_11(y))  # (B, 1024, 256)
        # y = self.maxPool_12(y)   # (B, 1024, 128)
        # y = self.batchNorm_13(y)
    
        jy = self.cnn_j(y)  # (B, 128, 128)
        ky = self.cnn_k(y)  # (B, 128, 128)
        ly = self.cnn_l(y)  # (B, 1024, 128)

        E = torch.bmm(jy.permute(0, 2, 1), ky)  # (B, 128, 128)
        
        E = E.view(-1).softmax(0).view(*E.shape)
        A = torch.bmm(ly, E)    # (B, 1024, 128)

        context_vector = y + self.gamma * A     # (B, 1024, 128)

        return context_vector       # (B, 1024, 128)

class BertEmbed(torch.nn.Module):
    def __init__(self):
        ### sentence를 받아서 BERT를 통해 벡터화
        super(BertEmbed, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bert = BertModel.from_pretrained('bert-large-uncased', output_hidden_states = True)
        
        self.update_num = 0

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        token = self.tokenizer(x, add_special_tokens=True, padding=True)

        input_tensor = torch.tensor(token.data['input_ids']).to(device)
        token_type_ids_tensor = torch.tensor(token.data['token_type_ids']).to(device)
        attention_mask_tensor = torch.tensor(token.data['attention_mask']).to(device)
        
        ##################
        bert_x = self.bert(input_tensor,
                            token_type_ids=token_type_ids_tensor,
                            attention_mask=attention_mask_tensor,)
        '''if self.update_num < 4:
            bert_x = self.bert(input_tensor,
                            token_type_ids=token_type_ids_tensor,
                            attention_mask=attention_mask_tensor,)
        else:
            with torch.no_grad():
                bert_x = self.bert(input_tensor,
                                token_type_ids=token_type_ids_tensor,
                                attention_mask=attention_mask_tensor,)'''
            
        token_hidden = torch.stack(bert_x[2], dim=0).permute(0, 1, 2, 3)
        sentences_embed = torch.mean(token_hidden[-2], dim=1)
        
        self.update_num += 1

        return sentences_embed