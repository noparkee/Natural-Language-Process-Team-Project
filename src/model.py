# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from network import AudioFeaturizer, BertEmbed, EmotionClassifier

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
        
        self.text_projection = nn.Linear(768, 256)      # 256
        self.audio_projection = nn.Linear(768, 256)     # 256
        
        self.num_classes = num_classes
        self.classifier = nn.Linear(512, self.num_classes)      # 512

        self.optimizer = get_optimizer(self.parameters())


    def update(self, minibatch):
        # list, tensor, list, list 
        text, mfcc, mfcc_len, label = minibatch
        
        cls_loss = 0

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(torch.squeeze(audio_embed, dim=0))
        features = torch.cat((text_features, audio_features), dim=1)
        
        cls_outputs = self.classifier(features)
        cls_loss = F.cross_entropy(cls_outputs, label)

        '''
        threshold 이상이 아니면 xxx로 예측하도록..? <-- 이 부분은 학습할 때는 상관없고 evaluate할 때 필요해보이는데
        '''

        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.5 * sd

        loss = cls_loss + sd_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ###
        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        total = float(len(text))
        
        #return {'loss': loss.item()}
        return correct, total, cls_loss.item()

        
        
    def evaluate(self, minibatch):
        text, mfcc, mfcc_len, label = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        #with torch.no_grad():
        text_features = self.text_projection(text_embed)
        audio_features = torch.squeeze(self.audio_projection(audio_embed), dim=0)
        features = torch.cat((text_features, audio_features), dim=1)
        
        cls_outputs = self.classifier(features)
        
        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        total = float(len(text))

        cls_loss = F.cross_entropy(cls_outputs, label)
        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.1 * sd
        loss = cls_loss + sd_loss

        return correct, total, loss.item()
