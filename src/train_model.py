# -*- coding: utf-8 -*-
# +
import sys
import random
import numpy as np
import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader

from data import get_data_iterators, set_device
from model import AudioTextModel

import os
import time
# -

## Hyperparameters
ITER = 100
NUM_CLASSES = 6
BATCH_SIZE = 32

## global seed 고정
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''parser = argparse.ArgumentParser()
parser.add_argument("--concat", type=bool, default=True)
args = parser.parse_args()
CONCAT = args.concat'''

# ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('### device: ' + str(device))
print("### cpu num: ", os.cpu_count())
model = AudioTextModel(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES).to(device)
print("### load model")

train, test = get_data_iterators()
print("### load data")

print("### start train")
print("\n")
for step in range(ITER):   # epoch
        
    starttime = time.time()

    model.debuglst_test = torch.zeros(6, 6)
    model.debuglst_train = torch.zeros(6, 6)

    train_correct_lst, train_loss_lst = [], []
    #text_correct_lst, audio_correct_lst, correct_lst, text_loss_lst, audio_loss_lst, loss_lst = [], [], [], [], [], []
    for batch_idx, minibatch in enumerate(train):       # 한 번의 epoch                 
        minibatch = set_device(minibatch, device)
        correct, loss = model.update(minibatch)
        #text_correct, audio_correct, correct, text_loss, audio_loss, loss = model.update(minibatch)
        
        '''text_correct_lst.append(text_correct)
        audio_correct_lst.append(audio_correct)
        correct_lst.append(correct)
        text_loss_lst.append(text_loss)
        audio_loss_lst.append(audio_loss)
        loss_lst.append(loss)'''

        train_correct_lst.append(correct)
        train_loss_lst.append(loss)
    '''print("train===")
    print('# step [{}/{}], text loss: {}'.format(step + 1, ITER, (sum(text_loss_lst)/len(text_loss_lst))))
    print('# step [{}/{}], audio loss: {}'.format(step + 1, ITER, (sum(audio_loss_lst)/len(audio_loss_lst))))
    print('# step [{}/{}], train loss: {}'.format(step + 1, ITER, (sum(loss_lst)/len(loss_lst))))
    print('# step [{}/{}], text accuracy: {}'.format(step + 1, ITER, (sum(text_correct_lst)/len(text_correct_lst) / BATCH_SIZE)))
    print('# step [{}/{}], audio accuracy: {}'.format(step + 1, ITER, (sum(audio_correct_lst)/len(audio_correct_lst) / BATCH_SIZE)))
    print('# step [{}/{}], train accuracy: {}'.format(step + 1, ITER, (sum(correct_lst)/len(correct_lst) / BATCH_SIZE)))'''

    test_correct_lst, test_loss_lst = [], []
    #text_correct_lst, audio_correct_lst, correct_lst, text_loss_lst, audio_loss_lst, loss_lst = [], [], [], [], [], []
    for batch_idx, minibatch in enumerate(test):
        minibatch = set_device(minibatch, device)
        with torch.no_grad():
            correct, loss = model.evaluate(minibatch)
            #text_correct, audio_correct, correct, text_loss, audio_loss, loss = model.evaluate(minibatch)
        
        '''text_correct_lst.append(text_correct)
        audio_correct_lst.append(audio_correct)
        correct_lst.append(correct)
        text_loss_lst.append(text_loss)
        audio_loss_lst.append(audio_loss)
        loss_lst.append(loss)'''
        
        test_correct_lst.append(correct)
        test_loss_lst.append(loss)

        
    
    #if (step+1) % 10 == 0:   # 10 epoch 마다 진행상황
    #    print('# step [{}/{}], loss: {}'.format(step + 1, ITER, (sum(loss_lst)/len(loss_lst))))
    
    '''print("test===")
    print('# step [{}/{}], text loss: {}'.format(step + 1, ITER, (sum(text_loss_lst)/len(text_loss_lst))))
    print('# step [{}/{}], audio loss: {}'.format(step + 1, ITER, (sum(audio_loss_lst)/len(audio_loss_lst))))
    print('# step [{}/{}], test loss: {}'.format(step + 1, ITER, (sum(loss_lst)/len(loss_lst))))
    print('# step [{}/{}], text accuracy: {}'.format(step + 1, ITER, (sum(text_correct_lst)/len(text_correct_lst) / BATCH_SIZE)))
    print('# step [{}/{}], audio accuracy: {}'.format(step + 1, ITER, (sum(audio_correct_lst)/len(audio_correct_lst) / BATCH_SIZE)))
    print('# step [{}/{}], test accuracy: {}'.format(step + 1, ITER, (sum(correct_lst)/len(correct_lst) / BATCH_SIZE)))'''

    print('# step [{}/{}], train loss: {}'.format(step + 1, ITER, (sum(train_loss_lst)/len(train_loss_lst))))
    print('# step [{}/{}], test loss: {}'.format(step + 1, ITER, (sum(test_loss_lst)/len(test_loss_lst))))
    print('# step [{}/{}], train accuracy: {}'.format(step + 1, ITER, (sum(train_correct_lst)/len(train_correct_lst) / BATCH_SIZE)))
    print('# step [{}/{}], test accuracy: {}'.format(step + 1, ITER, (sum(test_correct_lst)/len(test_correct_lst) / BATCH_SIZE)))
    print("delta: ", (time.time() - starttime), "sec")

    print(model.debuglst_train)
    print(model.debuglst_test)
    print("==========")
    

    '''
    if step == 0 or (step+1) % 3 == 0:   # 30 epoch 마다 evaluate
        correct_lst = []
        with torch.no_grad():
            for batch_idx, minibatch in enumerate(test):  # full test data
                minibatch = set_device(minibatch, device)
                correct, loss = model.evaluate(minibatch)
                correct_lst.append(correct)
            #print(str(step+1) + " epoch, eval: " + str(((sum(correct_lst)/len(correct_lst)) / 32).item()))
            print('# step [{}/{}], accuracy: {}'.format(step + 1, ITER, ((((sum(correct_lst)/len(correct_lst)) / 32).item()))))
    '''
'''
# last
correct_lst = []
with torch.no_grad():
    for batch_idx, minibatch in enumerate(test):  # full test data
        minibatch = set_device(minibatch, device)
        correct, loss = model.evaluate(minibatch)
        correct_lst.append(correct)
    print("last epoch, eval: " + str(((sum(correct_lst)/len(correct_lst)) / 32).item()))
'''
