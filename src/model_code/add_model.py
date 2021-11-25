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
    def __init__(self, batch_size, num_classes):
        super(AudioTextModel, self).__init__()
        self.textEmbedding = BertEmbed()
        self.audioEmbedding = AudioFeaturizer()
        
        self.text_projection = nn.Linear(768, 256)      # 256
        self.audio_projection = nn.Linear(2048, 256)     # 256
        
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.linear_layer = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, self.num_classes)      # 512

        self.lambda1 = nn.parameter.Parameter(torch.ones(1))

        self.vlayer1 = nn.Linear(256, 128)
        self.vlayer2 = nn.Linear(128, 1)
        self.alayer1 = nn.Linear(256, 128)
        self.alayer2 = nn.Linear(128, 1)
        self.dlayer1 = nn.Linear(256, 128)
        self.dlayer2 = nn.Linear(128, 1)

        self.optimizer = get_optimizer(self.parameters())


    def update(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(torch.squeeze(audio_embed, dim=0))
        features = (1-self.lambda1) * text_features + self.lambda1 * audio_features
        #features = self.lambda1 * text_features + (1-self.lambda1) * audio_features

        cls_outputs = self.classifier(features)
        cls_loss = F.cross_entropy(cls_outputs, label)
        
        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.5 * sd

        predict_v1 = F.relu(self.vlayer1(features))
        predict_a1 = F.relu(self.alayer1(features))
        predict_d1 = F.relu(self.dlayer1(features))
        predict_v2 = torch.squeeze(self.vlayer2(predict_v1), dim=1)
        predict_a2 = torch.squeeze(self.vlayer2(predict_a1), dim=1)
        predict_d2 = torch.squeeze(self.vlayer2(predict_d1), dim=1)

        v_loss = F.mse_loss(predict_v2, v)
        a_loss = F.mse_loss(predict_a2, a)
        d_loss = F.mse_loss(predict_d2, d)

        loss = cls_loss + sd_loss + 0.5*(v_loss + a_loss + d_loss)
        
        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return correct, loss.item()
        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()

        
        
    def evaluate(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        
        text_features = self.text_projection(text_embed)
        audio_features = torch.squeeze(self.audio_projection(audio_embed), dim=0)
        features = (1-self.lambda1) * text_features + self.lambda1 * audio_features

        
        cls_outputs = self.classifier(features)
        cls_loss = F.cross_entropy(cls_outputs, label)
        
        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.1 * sd

        predict_v1 = F.relu(self.vlayer1(features))
        predict_a1 = F.relu(self.alayer1(features))
        predict_d1 = F.relu(self.dlayer1(features))
        predict_v2 = torch.squeeze(self.vlayer2(predict_v1), dim=1)
        predict_a2 = torch.squeeze(self.vlayer2(predict_a1), dim=1)
        predict_d2 = torch.squeeze(self.vlayer2(predict_d1), dim=1)

        v_loss = F.mse_loss(predict_v2, v)
        a_loss = F.mse_loss(predict_a2, a)
        d_loss = F.mse_loss(predict_d2, d)

        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        total = float(len(text))

        loss = cls_loss + sd_loss + 0.5*(v_loss + a_loss + d_loss)

        #return correct, total, loss.item()
        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()
        return correct, loss.item()
