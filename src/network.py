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

class AudioFeaturizer(torch.nn.Module):   # LSTM
    ### mfcc 벡터를 LSTM에 통과 시켜서 last hidden state return
    def __init__(self):
        super(AudioFeaturizer, self).__init__()
        
        self.lstm = nn.LSTM(input_size=20, hidden_size=2048, num_layers=2, batch_first=True)

    def forward(self, x, l):   # 여기서 x는 list of tensor list(tensor)
        
        #x = pad_sequence(x, batch_first=True, padding_value=0)
        x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        
        x, state = self.lstm(x)

        #return state[0]   # hidden state
        return state[0][1]   # hidden state

class BertEmbed(torch.nn.Module):
    def __init__(self):
        ### sentence를 받아서 BERT를 통해 벡터화
        super(BertEmbed, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        
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
