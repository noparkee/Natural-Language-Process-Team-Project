# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from network_code.network4 import AudioFeaturizer, BertEmbed

def get_optimizer(params):
    LEARNING_RATE = 0.0001x c
    WEIGHT_DECAY = 0
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return optimizer


class AudioTextModel(torch.nn.Module):
    def __init__(self, batch_size, num_classes):
        super(AudioTextModel, self).__init__()
        self.textEmbedding = BertEmbed()
        self.audioEmbedding = AudioFeaturizer()
        
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.text_classifier = nn.Linear(1024, self.num_classes)
        self.audio_classifier = nn.Linear(510 * 128, self.num_classes)

        self.lambda1 = nn.parameter.Parameter(torch.ones(1))      # for classifier

        self.optimizer = get_optimizer(self.parameters())


    def update(self, minibatch):
        # list, tensor, list, list 
        text, mfcc, mfcc_len, label, v, a, d = minibatch
        
        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = text_embed
        audio_features = audio_embed.reshape(self.batch_size, 510 * 128)
        
        text_outputs = self.text_classifier(text_features)
        audio_outputs = self.audio_classifier(audio_features)
        
        outputs = (1-self.lambda1) * text_outputs + self.lambda1 * audio_outputs
        #outputs = audio_outputs
        
        cls_loss = F.cross_entropy(outputs, label)
        #cls_loss += (0.5 * (outputs ** 2).mean())

        loss = cls_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ###
        text_correct = (text_outputs.argmax(1).eq(label).float()).sum()
        audio_correct = (audio_outputs.argmax(1).eq(label).float()).sum()
        correct = (outputs.argmax(1).eq(label).float()).sum()
        
        for i in range(len(label)):
           self.debuglst_train[outputs.argmax(1)[i]][label[i]] += 1

        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()
        return correct, loss.item()

        
        
    def evaluate(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = text_embed
        audio_features = audio_embed.reshape(self.batch_size, 510 * 128)
        
        text_outputs = self.text_classifier(text_features)
        audio_outputs = self.audio_classifier(audio_features)
        
        outputs = (1-self.lambda1) * text_outputs + self.lambda1 * audio_outputs
        #outputs = audio_outputs
        
        cls_loss = F.cross_entropy(outputs, label)
        #cls_loss += (0.5 * (outputs ** 2).mean())

        loss = cls_loss
        
        correct = (outputs.argmax(1).eq(label).float()).sum()

        
        #debug correct list
        for i in range(len(label)):
           self.debuglst_test[outputs.argmax(1)[i]][label[i]] += 1

        #return correct, total, loss.item()
        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()
        return correct, loss.item()
