# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.networks import AudioFeaturizer, BertEmbed, EmotionClassifier

def get_optimizer(params):
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return optimizer


class AudioTextModel(torch.nn.Module):
    def __init(self, num_classes):
        super(AudioTextModel, self).__init__()
        self.textEmbedding = BertEmbed()
        self.audioEmbedding = AudioFeaturizer()
        
        self.text_projection = nn.Linear(768, 512)
        self.audio_projection = nn.Linear(768, 512)
        
        self.classifier = nn.Linear(1024, num_classes)

        self.num_classes = num_classes
        
        self.optimizer = get_optimizer(self.parameters())


    def update(self, minibatch):
        # mfcc: list of tensor
        text, mfcc, label = x
        
        cls_loss = 0
        
        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(audio_embed)
        features = torch.cat((text_features, audio_features), dim=1)
        
        cls_outputs = self.classifier(features)
        cls_loss = F.cross_entropy(cls_outputs, label)
        ###cls_loss = F.softmax(cls_outputs, dim=1)
        
        self.optimizer.zero_grad()
        cls_loss.backward()
        self.optimizer.step()
        
        return OrderedDict({'cls_loss': cls_loss })

        
        
    def evaluate(self, minibatch):
        text, mfcc, label = minibatch  # 추후에 text는 ASR로 변경 일단은 주어진 transcript 이용

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(audio_embed)
        features = torch.cat((text_features, audio_features), dim=1)
        
        cls_outputs = self.classifier(features)
        
        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        
        total = float(len(x))

        return correct, total



