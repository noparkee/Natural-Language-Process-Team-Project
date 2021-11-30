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

        self.attention_size = 512       # 512
        self.hidden1 = 512             # 512
        self.hidden2 = 1024             # 1024

        self.lstm1 = nn.LSTM(input_size=20, hidden_size=self.hidden1, num_layers=1, batch_first=True, bidirectional=True)     # 양방향
        self.lstm2 = nn.LSTM(input_size=2*self.hidden1, hidden_size=self.hidden2, num_layers=1, batch_first=True, bidirectional=True)     # 양방향

        self.wQ = nn.Linear(2*self.hidden1, self.attention_size)
        self.wK = nn.Linear(2*self.hidden1, self.attention_size)
        self.wV = nn.Linear(2*self.hidden1, self.attention_size)

        
    def forward(self, x, l):
        n_batchsize, ml, vector_dim = x.size()

        #x = pad_sequence(x, batch_first=True, padding_value=0)
        x1 = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        x1, state1 = self.lstm1(x1)
        x, _ = pad_packed_sequence(x1, total_length=max(l), batch_first=True)        # x: (B, ml, 2*hidden1)
        
        ### attention (bahdanau)
        #hidden = torch.cat((state[0][2], state[0][3]), dim=1)       # get hidden state / 근데 여기서 index가 2와 3이 맞는지는 잘 모르겠음
        #score = self.V(torch.tanh(self.W1(x) + self.W2(torch.unsqueeze(hidden, dim=1))))
        #attention_weights = F.softmax(score)        # (B, Max_L, 1)
        #context_vector = torch.squeeze(torch.bmm(attention_weights.permute(0, 2, 1), x), dim=1)

        '''### self attention
        query = self.wQ(x)      # (B, ml, attenion_size)
        key = self.wK(x)
        value = self.wV(x)

        querykey = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(self.attention_size)
        sfmx = F.softmax(querykey, dim=-1)
        attnetion_value = torch.bmm(sfmx, value)            # (B, ml, attntion_size)
        
        x2 = pack_padded_sequence(attnetion_value, l, batch_first=True, enforce_sorted=False)
        x2, state2 = self.lstm2(x)
    
        #return state[0]   # hidden state
        #return torch.cat((state2[0][2], state2[0][3]), dim=1)       # get hidden state / 근데 여기서 index가 2와 3이 맞는지는 잘 모르겠음
        return torch.cat((state2[0][0], state2[0][1]), dim=1)'''
        return torch.cat((state1[0][0], state1[0][1]), dim=1)


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
        
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same')
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')

        self.batchnorm2d_2 = nn.BatchNorm2d(32) ##수정했음!
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv2d_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv2d_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')

        self.batchnorm2d_5 = nn.BatchNorm2d(256) ##수정했음!
        self.conv2d_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.conv2d_7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2)

        self.batchnorm2d_7 = nn.BatchNorm2d(1024) ##수정했음!

        self.cnn_j = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1)
        self.cnn_k = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=1)
        self.cnn_l = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)

        self.gamma = nn.parameter.Parameter(torch.zeros(1))

        

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):   
        
        x = self.conv2d_1(x)                                    # (16, 448, 448)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)      # (16, 224, 224)
        x = self.conv2d_2(x)                                    # (32, 224, 224)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)      # (32, 112, 112)

        x = self.batchnorm2d_2(x)   ##수정했음!

        x = self.conv2d_3(x)                                    # (64, 112, 112)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)      # (64, 56, 56)
        x = self.conv2d_4(x)                                    # (128, 56, 56)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)
        x = self.conv2d_5(x)                                    # (256, 28, 28)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)

        x = self.batchnorm2d_5(x)   ##수정했음!                  # (256, 14, 14)
        size = x.size()
        
###
        x = x.reshape(size[0], size[1], -1)                     # (256, 196)

        # x = torch.squeeze(x)

        jx = self.cnn_j(x)
        kx = self.cnn_k(x)
        lx = self.cnn_l(x)

        E = torch.bmm(jx.permute(0, 2, 1), kx)
        E = E.view(-1).softmax(0).view(*E.shape)
        A = torch.bmm(lx, E) 

        x = x + self.gamma * A

        x = x.reshape(size[0], size[1], size[2], size[3])       # (256, 14, 14)
###

        x = self.conv2d_6(x)                                    # (512, 6, 6)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2, stride=2, input=x)      # (512, 3, 3)
        x = self.conv2d_7(x)                       # (1024, 1, 1)
        x = F.relu(x)

        x = self.batchnorm2d_7(x)   ##수정했음!     # (1024, 1, 1)

        x = x.squeeze()             # (1024)
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
        