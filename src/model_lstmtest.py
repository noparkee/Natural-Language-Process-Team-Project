# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from network import AudioFeaturizer, BertEmbed
from network2 import AudioFeaturizer2

def get_optimizer(params):
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return optimizer


class AudioTextModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(AudioTextModel, self).__init__()
        self.textEmbedding = BertEmbed()
        self.audioEmbedding = AudioFeaturizer()
        
        self.text_projection = nn.Linear(1024, 256)      # 256
        self.audio_projection = nn.Linear(1024, 256)     # 256
        
        self.num_classes = num_classes
        
        self.linear_layer1 = nn.Linear(512, 256)
        self.linear_layer2 = nn.Linear(256, 128)

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, dropout=0.5)
        self.classifier = nn.Linear(512, self.num_classes)      # 512

        self.vlayer = nn.Linear(128, 1)
        self.alayer = nn.Linear(128, 1)
        self.dlayer = nn.Linear(128, 1)

        self.optimizer = get_optimizer(self.parameters())


    def update(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(torch.squeeze(audio_embed, dim=0))
        features = torch.cat((text_features, audio_features), dim=1).unsqueeze(1)
        features, state = self.lstm(features)

        cls_outputs = self.classifier(state[0][1])
        cls_loss = F.cross_entropy(cls_outputs, label)
        
        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.5 * sd

        # predict_v = torch.squeeze(self.vlayer(features), dim=1)
        # predict_a = torch.squeeze(self.alayer(features), dim=1)
        # predict_d = torch.squeeze(self.dlayer(features), dim=1)

        #v_loss = F.mse_loss(predict_v, v)
        #a_loss = F.mse_loss(predict_a, a)
        #d_loss = F.mse_loss(predict_d, d)

        loss = cls_loss + sd_loss #+ 0.5*(v_loss + a_loss + d_loss)
        
        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #return {'loss': loss.item()}
        return correct, loss.item()
        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()

        
        
    def evaluate(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(torch.squeeze(audio_embed, dim=0))
        features = torch.cat((text_features, audio_features), dim=1).unsqueeze(1)
        features, state = self.lstm(features)

        cls_outputs = self.classifier(state[0][1])
        cls_loss = F.cross_entropy(cls_outputs, label)
        
        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.1 * sd

        # predict_v = torch.squeeze(self.vlayer(features), dim=1)
        # predict_a = torch.squeeze(self.alayer(features), dim=1)
        # predict_d = torch.squeeze(self.dlayer(features), dim=1)
        # v_loss = F.mse_loss(predict_v, v)
        # a_loss = F.mse_loss(predict_a, a)
        # d_loss = F.mse_loss(predict_d, d)

        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        total = float(len(text))

        loss = cls_loss + sd_loss #+ 0.5*(v_loss + a_loss + d_loss)

        #return correct, total, loss.item()
        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()
        return correct, loss.item()
