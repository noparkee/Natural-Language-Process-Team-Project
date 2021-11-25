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
        
        self.text_projection = nn.Linear(1024, 512)
        self.audio_projection = nn.Linear(1024, 512)
        
        self.num_classes = num_classes
        
        self.text_classifier = nn.Linear(512, self.num_classes)
        self.audio_classifier = nn.Linear(512, self.num_classes)

        self.text_vlayer = nn.Linear(512, 1)
        self.text_alayer = nn.Linear(512, 1)
        self.text_dlayer = nn.Linear(512, 1)
        self.audio_vlayer = nn.Linear(512, 1)
        self.audio_alayer = nn.Linear(512, 1)
        self.audio_dlayer = nn.Linear(512, 1)

        self.lambda1 = nn.parameter.Parameter(torch.Tensor(1))      # for classifier
        self.lambdav = nn.parameter.Parameter(torch.Tensor(1))      # for v
        self.lambdaa = nn.parameter.Parameter(torch.Tensor(1))      # for a
        self.lambdad = nn.parameter.Parameter(torch.Tensor(1))      # for d

        self.optimizer = get_optimizer(self.parameters())


    def update(self, minibatch):
        # list, tensor, list, list 
        text, mfcc, mfcc_len, label, v, a, d = minibatch
        
        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(torch.squeeze(audio_embed, dim=0))
        
        text_outputs = self.text_classifier(text_features)
        audio_outputs = self.audio_classifier(audio_features)
        outputs = self.lambda1 * text_outputs + (1-self.lambda1) * audio_outputs
        
        cls_loss = F.cross_entropy(text_outputs, label) + F.cross_entropy(audio_outputs, label) + F.cross_entropy(outputs, label)
        #cls_loss += (0.5 * (outputs ** 2).mean())
        
        text_v = torch.squeeze(self.text_vlayer(text_features), dim=1)
        text_a = torch.squeeze(self.text_alayer(text_features), dim=1)
        text_d = torch.squeeze(self.text_dlayer(text_features), dim=1)
        audio_v = torch.squeeze(self.audio_vlayer(audio_features), dim=1)
        audio_a = torch.squeeze(self.audio_alayer(audio_features), dim=1)
        audio_d = torch.squeeze(self.audio_dlayer(audio_features), dim=1)

        v_loss = F.mse_loss(text_v, v) + F.mse_loss(audio_v, v) + F.mse_loss(self.lambdav * text_v + (1-self.lambdav) * audio_v, v)
        a_loss = F.mse_loss(text_a, a) + F.mse_loss(audio_a, a) + F.mse_loss(self.lambdaa * text_a + (1-self.lambdaa) * audio_a, a)
        d_loss = F.mse_loss(text_d, d) + F.mse_loss(audio_d, d) + F.mse_loss(self.lambdad * text_d + (1-self.lambdad) * audio_d, d)


        loss = cls_loss + v_loss + a_loss + d_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ###
        text_correct = (text_outputs.argmax(1).eq(label).float()).sum()
        audio_correct = (audio_outputs.argmax(1).eq(label).float()).sum()
        correct = (outputs.argmax(1).eq(label).float()).sum()
        
        return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()

        
        
    def evaluate(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(torch.squeeze(audio_embed, dim=0))
        
        text_outputs = self.text_classifier(text_features)
        audio_outputs = self.audio_classifier(audio_features)
        outputs = self.lambda1 * text_outputs + (1-self.lambda1) * audio_outputs
        
        cls_loss = F.cross_entropy(text_outputs, label) + F.cross_entropy(audio_outputs, label) + F.cross_entropy(outputs, label)
        #cls_loss += (0.5 * (outputs ** 2).mean())
        
        text_v = torch.squeeze(self.text_vlayer(text_features), dim=1)
        text_a = torch.squeeze(self.text_alayer(text_features), dim=1)
        text_d = torch.squeeze(self.text_dlayer(text_features), dim=1)
        audio_v = torch.squeeze(self.audio_vlayer(audio_features), dim=1)
        audio_a = torch.squeeze(self.audio_alayer(audio_features), dim=1)
        audio_d = torch.squeeze(self.audio_dlayer(audio_features), dim=1)

        v_loss = F.mse_loss(text_v, v) + F.mse_loss(audio_v, v) + F.mse_loss(self.lambdav * text_v + (1-self.lambdav) * audio_v, v)
        a_loss = F.mse_loss(text_a, a) + F.mse_loss(audio_a, a) + F.mse_loss(self.lambdaa * text_a + (1-self.lambdaa) * audio_a, a)
        d_loss = F.mse_loss(text_d, d) + F.mse_loss(audio_d, d) + F.mse_loss(self.lambdad * text_d + (1-self.lambdad) * audio_d, d)
        
        text_correct = (text_outputs.argmax(1).eq(label).float()).sum()
        audio_correct = (audio_outputs.argmax(1).eq(label).float()).sum()
        correct = (outputs.argmax(1).eq(label).float()).sum()

        #return correct, total, loss.item()
        return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()
