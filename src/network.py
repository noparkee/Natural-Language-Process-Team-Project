# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizer, BertModel

class AudioFeaturizer(torch.nn.Module):
    def __init__(self, projection_size):        # projection size 채워야해
        super(AudioFeaturizer, self).__init__()
        
        ### 나중에 여유되면 config
        self.lstm = nn.LSTM(4020, 768, batch_first=True)

    def forward(self, x):
        x, state = self.lstm(x)
        #a_feature = self.linear(state[0])

        return state[0]

class BertEmbed(torch.nn.Module):
    def __init__(self):
        super(BertEmbed, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

        def forward(self, x):
            token = self.tokenizer.encode(x, add_special_tokens=True, padding=True)
            
            input_tensor = torch.tensor(token.data['input_ids'])
            token_type_ids_tensor = torch.tensor(token.data['token_type_ids'])
            attention_mask_tensor = torch.tensor(token.data['attention_mask'])

            bert_x = self.bert(input_tensor,
                        token_type_ids=token_type_ids_tensor,
                        attention_mask=attention_mask_tensor,)
            
            token_hidden = torch.stack(bert_x[2], dim=0).permute(0, 1, 2, 3)
            sentences_embed = torch.mean(token_hidden[-2], dim=1)

            return sentences_embed

class EmotionClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        
        self.linear1 = nn.Linear(1024, 256)
        self.linear2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        
        sm = F.softmax(x, dim=1)
