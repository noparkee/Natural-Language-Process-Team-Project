# -*- coding: utf-8 -*-
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

word = pd.read_pickle('../data' + '/description.pkl')
label_words = word['label'].unique()
label_num = [0,1,2,3,4,5,6,7,8,9,10]
word


### Dataset 상속
class ourDataset(Dataset): 
    def __init__(self):
    ##data set 경로(루트경로)
        self.data_path = '../data'

        audio = pd.read_pickle(self.data_path + '/audio_vec.pkl')
        word = pd.read_pickle(self.data_path + '/description.pkl')

        ##true y's
        self.sentence = word['sentence']
        #self.label = word['label']
        self.label = word['label_num']

        ##audio feature
        #self.mfcc = audio['mfcc']
        
        #self.mfcc_tensor = torch.nn.utils.rnn.pad_sequence(audio['mfcc_tensor'].tolist(),batch_first=True, padding_value=0)
        
        self.mfcc_tensor = audio['mfcc_tensor'].tolist()
        self.mfcc_len = audio['len'].tolist()

        ##v,a,d
        self.v = word['v']
        self.a = word['a']
        self.d = word['d']
    
  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.label)
    
  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index): 
        sentence = self.sentence[index]
        audio_embed = self.mfcc_tensor[index]
        audio_len = self.mfcc_len[index]
        label = self.label[index]
    
        return sentence, audio_embed, audio_len, label


def collate_fn(batch):
    sentence, audio_embedt, audio_len, label = zip(*batch)
    
    sentence = list(sentence)
    #audio_embed = torch.stack(audio_embed, 0)
    
    #batch 단위로 padding
    audio_embed = torch.nn.utils.rnn.pad_sequence(audio_embedt, batch_first=True)
    audio_len = list(audio_len)
    label = torch.tensor(label)
    
    return sentence, audio_embed, audio_len, label

def get_data_iterators():
    BATCH_SIZE = 32
    
    dataset = ourDataset()
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [6000, 4039])
  
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True, collate_fn=collate_fn)

    return train_loader, test_loader


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


