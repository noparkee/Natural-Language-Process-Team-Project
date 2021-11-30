# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# word = pd.read_pickle('../data' + '/description.pkl')
# label_words = word['label'].unique()
# label_num = [0,1,2,3,4,5,6,7,8,9,10]
# word


#word = pd.read_pickle('../data' + '/description.pkl')
#label_words = word['label'].unique()
#label_num = [0,1,2,3,4,5,6,7,8,9,10]
#word


### Dataset 상속
class ourDataset(Dataset): 
    def __init__(self):
        ##data set 경로(루트경로)
        self.data_path = '../data'

        #audio = pd.read_pickle(self.data_path + '/audio_vec4_44100.pkl')
        #description = pd.read_pickle(self.data_path + '/description4.pkl')
        data = pd.read_pickle(self.data_path + '/data3.pkl')

        ##true y's
        self.sentence = data['sentence']
        #self.label = description['label']
        self.label = data['label_num']

        ##audio feature
        #self.mfcc = audio['mfcc']
        
        #self.mfcc_tensor = torch.nn.utils.rnn.pad_sequence(audio['mfcc_tensor'].tolist(),batch_first=True, padding_value=0)
        
        self.mfcc_tensor = data['mfcc_tensor'].tolist()
        self.mfcc_len = data['len'].tolist()

        ##v,a,d
        self.v = data['v'].astype(float).tolist()
        self.a = data['a'].astype(float).tolist()
        self.d = data['d'].astype(float).tolist()

        ### image
        self.image_path = data['image_path']
        self.transform = get_transforms()

    ### 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.label)
    
    ### 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index): 
        sentence = self.sentence[index]
        audio_embed = self.mfcc_tensor[index]
        audio_len = self.mfcc_len[index]
        label = self.label[index]

        v = self.v[index]
        a = self.a[index]
        d = self.d[index]

        ###
        image_path = self.image_path[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # return sentence, audio_embed, audio_len, label, v, a, d
        return sentence, audio_embed, audio_len, label, v, a, d, image


def collate_fn(batch):
    ## zip: 튜플의 리스트를 리스트의 튜플로 바꿔줌
    # sentence, audio_embedt, audio_len, label,  v, a, d = zip(*batch)
    sentence, audio_embedt, audio_len, label, v, a, d, images = zip(*batch)
    
    sentence = list(sentence)
    #audio_embed = torch.stack(audio_embed, 0)
    
    #batch 단위로 padding
    audio_embed = torch.nn.utils.rnn.pad_sequence(audio_embedt, batch_first=True)
    audio_len = list(audio_len)
    label = torch.tensor(label)
    
    v = torch.tensor(v)
    a = torch.tensor(a)
    d = torch.tensor(d)

    ###
    images = torch.stack(images, 0)

    # return sentence, audio_embed, audio_len, label, v, a, d
    return sentence, audio_embed, audio_len, label, v, a, d, images

def get_data_iterators(BATCH_SIZE):
    ## hyperparameters
    NUM_WORKERS = 12
    DROP_LAST = True
    SHUFFLE = False
    
    ## Total 7487 / 10039
    dataset = ourDataset()

    TOTAL = dataset.__len__()
    NUM_TRAIN = int(TOTAL * 0.8)
    NUM_TEST = TOTAL - NUM_TRAIN

    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [NUM_TRAIN, NUM_TEST])
  
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, drop_last=DROP_LAST, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, drop_last=DROP_LAST, collate_fn=collate_fn)

    return train_loader, test_loader


def get_transforms():
    """ get transforms for CUB datasets """
    resize, cropsize = 512, 448

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
