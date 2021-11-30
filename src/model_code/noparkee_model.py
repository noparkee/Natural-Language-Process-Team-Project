# -*- coding: utf-8 -*-
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from network import AudioFeaturizer, BertEmbed, ResNet, ImageFeaturizer

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
        self.imageEmbedding = ImageFeaturizer()

        self.text_projection = nn.Linear(1024, 512)      # 512
        self.audio_projection = nn.Linear(2048, 512)     # 512
        self.image_projection = nn.Linear(1024, 256)     # 256

        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(1024, self.num_classes)         # 1024+256

        self.optimizer = get_optimizer(self.parameters())

        ###
        #self.vlayer = nn.Linear(1024+256, 1)
        #self.alayer = nn.Linear(1024+256, 1)
        #self.dlayer = nn.Linear(1024+256, 1)

        #self.query_conv = nn.Conv2d(in_channels = 256, out_channels = 256//8, kernel_size=1)
        #self.key_conv  = nn.Conv2d(in_channels = 256, out_channels = 256//8, kernel_size=1) 
        #self.value_conv   = nn.Conv2d(in_channels = 256, out_channels = 1024, kernel_size=1)         
        #self.gamma = nn.Parameter(torch.zeros(1))
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    '''def make_image_features(self, x):
        # SAGAN
        n_batchsize, n_channel, width, height = x.size()        # (B, 256, 28, 28) 

        proj_query = self.query_conv(image_before_pooling).view(n_batchsize,-1,width*height).permute(0,2,1)        # B X C X (N)
        print(proj_query.size())
        proj_key = self.key_conv(image_before_pooling).view(n_batchsize,-1,width*height)                           # B X C x (*W*H)
        print(proj_key.size())
        energy = torch.bmm(proj_query,proj_key)                                                                    # transpose check
        print(energy.size())
        attention = F.softmax(energy, dim=-1)            #####################                                           # BX (N) X (N)          # (32, 196, 196)
        print(attention.size())
                
        proj_value = self.value_conv(image_before_pooling).view(n_batchsize,-1,width*height)                        # B X C X N
        print(proj_value.size())
        
        out = torch.bmm(proj_value,attention.permute(0,2,1))        # (32, 2048, 196)
        print(out.size())
        out = out.view(n_batchsize,n_channel,width,height)          # (32, 2048, 14, 14)
        print(out.size())
        sum_out = self.gamma*out + image_before_pooling             # (32, 2048, 14, 14)
        print(sum_out.size())
        image_features = self.avgpool(sum_out).squeeze()            # (32, 2048)
        print(image_features.size())
        
        input()
        
        return image_features'''

    def update(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d, images = minibatch

        text_embed = self.textEmbedding(text)                   # (B, 1024)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)       # (B, 2*hidden)
        image_embed = self.imageEmbedding(images)

        text_features = self.text_projection(text_embed)
        audio_features = self.audio_projection(torch.squeeze(audio_embed, dim=0))
        image_features = self.image_projection(image_embed)
        #features = torch.cat((text_features, audio_features, image_features), dim=1)
        features = torch.cat((text_features, audio_features), dim=1)

        cls_outputs = self.classifier(features)
        #v_outputs = self.vlayer(features)
        #a_outputs = self.alayer(features)
        #d_outputs = self.dlayer(features)

        cls_loss = F.cross_entropy(cls_outputs, label)
        #v_loss = F.mse_loss(torch.squeeze(v_outputs, dim=1), v)
        #a_loss = F.mse_loss(torch.squeeze(a_outputs, dim=1), a)
        #d_loss = F.mse_loss(torch.squeeze(d_outputs, dim=1), d)

        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.5 * sd
        
        correct = (cls_outputs.argmax(1).eq(label).float()).sum()

        loss = cls_loss + sd_loss #+ 0.5*(v_loss + a_loss + d_loss)
        #+ F.cross_entropy(text_outputs, label) + F.cross_entropy(audio_outputs, label) + F.cross_entropy #+ 0.5*(v_loss + a_loss + d_loss)

        #debug
        for i in range(len(label)):
          self.debuglst_train[cls_outputs.argmax(1)[i]][label[i]] += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return correct, loss.item()
        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()

        
        
    def evaluate(self, minibatch):
        text, mfcc, mfcc_len, label, v, a, d, images = minibatch

        text_embed = self.textEmbedding(text)
        audio_embed = self.audioEmbedding(mfcc, mfcc_len)
        image_embed = self.imageEmbedding(images)
        
        text_features = self.text_projection(text_embed)
        audio_features = torch.squeeze(self.audio_projection(audio_embed), dim=0)
        image_features = self.image_projection(image_embed)
        #features = torch.cat((text_features, audio_features, image_features), dim=1)
        features = torch.cat((text_features, audio_features), dim=1)


        cls_outputs = self.classifier(features)
        #v_outputs = self.vlayer(features)
        #a_outputs = self.alayer(features)
        #d_outputs = self.dlayer(features)

        cls_loss = F.cross_entropy(cls_outputs, label)
        #v_loss = F.mse_loss(torch.squeeze(v_outputs, dim=1), v)
        #a_loss = F.mse_loss(torch.squeeze(a_outputs, dim=1), a)
        #d_loss = F.mse_loss(torch.squeeze(d_outputs, dim=1), d)
        
        sd = (cls_outputs ** 2).mean()
        sd_loss = 0.5 * sd

        correct = (cls_outputs.argmax(1).eq(label).float()).sum()
        #total = float(len(text))

        loss = cls_loss + sd_loss #+ 0.5*(v_loss + a_loss + d_loss)
        #F.cross_entropy(text_outputs, label) + F.cross_entropy(audio_outputs, label) + F.cross_entropy #+ 0.5*(v_loss + a_loss + d_loss)

        #debug correct list
        for i in range(len(label)):
           self.debuglst_test[cls_outputs.argmax(1)[i]][label[i]] += 1

        #return correct, total, loss.item()
        #return text_correct, audio_correct, correct, text_loss.item(), audio_loss.item(), loss.item()
        return correct, loss.item()
