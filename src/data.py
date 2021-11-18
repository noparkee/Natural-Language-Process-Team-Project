# -*- coding: utf-8 -*-
import torch
import pandas
from torch.utils.data import Dataset, DataLoader


# +
class ourDataset(Dataset):
    def __init__(self, train=True):
    label_list = [0,1,2,3,4]
    self.train = train
    num_data=1391
    num_train=1251
    num_test=140
    
    ##data set 경로(루트경로)
    self.data_path = '../data'
    
    audio = pd.read_pickle(self.data_path + '/audio_vec.pkl')
    word = pd.read_pickle(self.data_path + '/description.pkl')
    
    ##true y's
    self.sentence = word['sentence']
    self.label = word['label']
    
    ##audio feature
    self.mfcc = audio['mfcc']

    ##v,a,d
    self.v = word['v']
    self.a = word['a']
    self.d = word['d']
    
    def __getItem__(self, index):
        sentence = self.sentence[index]
        audio_embed = self.mfcc[index]
        label = self.label[index]
        
        
