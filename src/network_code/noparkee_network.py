# -*- coding: utf-8 -*-
import numpy as np
import pickle
import math

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
        
        hidden = 512
        self.lstm = nn.LSTM(input_size=20, hidden_size=hidden, num_layers=2, batch_first=True, bidirectional=True)     # 양방향

        #self.W1 = nn.Linear(hidden*2, hidden)
        #self.W2 = nn.Linear(hidden*2, hidden)
        #self.V = nn.Linear(hidden, 1)

        self.attention_size = 32
        self.wQ = nn.Linear(20, self.attention_size)
        self.wK = nn.Linear(20, self.attention_size)
        self.wV = nn.Linear(20, self.attention_size)


    def forward(self, x, l):
        n_batchsize, ml, vector_dim = x.size()

        #x = pad_sequence(x, batch_first=True, padding_value=0)
        x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        x, state = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max(l), batch_first=True)        # x: (B, ml, 2*hidden)
        
        ### attention (bahdanau)
        #hidden = torch.cat((state[0][2], state[0][3]), dim=1)       # get hidden state / 근데 여기서 index가 2와 3이 맞는지는 잘 모르겠음
        #score = self.V(torch.tanh(self.W1(x) + self.W2(torch.unsqueeze(hidden, dim=1))))
        #attention_weights = F.softmax(score)        # (B, Max_L, 1)
        #context_vector = torch.squeeze(torch.bmm(attention_weights.permute(0, 2, 1), x), dim=1)

        ### self attention
        query = self.wQ(x)      # (B, ml, attenion_size)
        key = self.wK(x)
        value = self.wV(x)

        querykey = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(self.attention_size)
        sfmx = F.softmax(querykey, dim=-1)
        attnetion_value = torch.bmm(sfmx, value)            # (B, ml, attntion_size)

        #return state[0]   # hidden state
        #return torch.cat((state[0][2], state[0][3]), dim=1)       # get hidden state / 근데 여기서 index가 2와 3이 맞는지는 잘 모르겠음
        return attnetion_value


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

        input_tensor = torch.tensor(token.data['input_ids']).to(device)                     # 단어 -> 숫자
        token_type_ids_tensor = torch.tensor(token.data['token_type_ids']).to(device)       # 문장 분류
        attention_mask_tensor = torch.tensor(token.data['attention_mask']).to(device)       # 패딩 분류용
        
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
        

        #token_hidden = torch.stack(bert_x[2], dim=0)
        #sentences_embed = torch.mean(token_hidden[-2], dim=1)

        # len(bert_x[2] = 25개의 layers
        # bert_x[2][layers][]
        #print(bert_x[2][5].size())  # (32, 51, 1024)
        #print(bert_x[2][5][0].size())
        #input()
        #sentences_embed = torch.cat((bert_x[2][1][:, -1], bert_x[2][1][:, -2], bert_x[2][1][:, -3], bert_x[2][1][:, -4]), dim=1)
        
        ### -1, -2, -3, -4 / -4, -3, -2, -1 : 둘 중에 뭐로 해야할까
        #sentences_embed = torch.cat((torch.mean(bert_x[2][-1], dim=1), torch.mean(bert_x[2][-2], dim=1), torch.mean(bert_x[2][-3], dim=1), torch.mean(bert_x[2][-4], dim=1)), dim=1)
        
        ### Embedding
        sentences_embed = torch.mean(bert_x[2][0], dim=1)

        self.update_num += 1

        return sentences_embed      # (B, 4*hidden)


class ImageFeaturizer(torch.nn.Module):   # LSTM
    ### mfcc 벡터를 LSTM에 통과 시켜서 last hidden state return
    def __init__(self):
        super(ImageFeaturizer, self).__init__()
        
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv2d_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.conv2d_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):   
        
        x = self.conv2d_1(x)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)
        #print(x.size())
        x = self.conv2d_2(x)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)
        #print(x.size())
        x = self.conv2d_3(x)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)
        #print(x.size())
        x = self.conv2d_4(x)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)
        #print(x.size())
        x = self.conv2d_5(x)
        x = F.relu(x)
        x = torch.squeeze(x)
        #print(x.size())
        #input()

        return x


class ResNet(torch.nn.Module):
    """ ResNet with the softmax chopped off and the batchnorm frozen """
    def __init__(self):
        super(ResNet, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048

        del self.network.fc
        self.network.fc = Identity()
        
        self.freeze_bn()        

    def forward(self, x):
        """ encode x into a feature vector of size n_outputs """
        return self.network(x)

    def train(self, mode=True):
        """ override the default train() to freeze the BN parameters """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class Identity(nn.Module):
    """ identity layer """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        