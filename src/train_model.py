# -*- coding: utf-8 -*-
# +
import sys
import random

import torch
import pandas as pd
from torch.utils.data import DataLoader

from data import get_data_iterators
from data import set_device
from model import AudioTextModel
# -

## Hyperparameters
ITER = 30   # 300
NUM_CLASSES = 6
BATCH_SIZE = 64

## global seed 고정
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# +
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioTextModel(num_classes=NUM_CLASSES).to(device)
print("### load model")

# +
train, test = get_data_iterators()
print("### load data")

# +
print("### start train")
print("\n")
for step in range(ITER):   # epoch
        
    train_correct_lst, train_loss_lst = [], []
    for batch_idx, minibatch in enumerate(train):       # 한 번의 epoch                 
        minibatch = set_device(minibatch, device)
        correct, total, loss = model.update(minibatch)
        train_correct_lst.append(correct)
        train_loss_lst.append(loss)

    test_correct_lst, test_loss_lst = [], []
    for batch_idx, minibatch in enumerate(test):  
        minibatch = set_device(minibatch, device)
        with torch.no_grad():
            correct, total, loss = model.evaluate(minibatch)
        test_correct_lst.append(correct)
        test_loss_lst.append(loss)
    
    #if (step+1) % 10 == 0:   # 10 epoch 마다 진행상황
    #    print('# step [{}/{}], loss: {}'.format(step + 1, ITER, (sum(loss_lst)/len(loss_lst))))
    
    print('# step [{}/{}], train loss: {}'.format(step + 1, ITER, (sum(train_loss_lst)/len(train_loss_lst))))
    print('# step [{}/{}], test loss: {}'.format(step + 1, ITER, (sum(test_loss_lst)/len(test_loss_lst))))
    print('# step [{}/{}], train accuracy: {}'.format(step + 1, ITER, ((((sum(train_correct_lst)/len(train_correct_lst)) / BATCH_SIZE).item()))))
    print('# step [{}/{}], test accuracy: {}'.format(step + 1, ITER, ((((sum(test_correct_lst)/len(test_correct_lst)) / BATCH_SIZE).item()))))
    print("==========")

    '''
    if step == 0 or (step+1) % 3 == 0:   # 30 epoch 마다 evaluate
        correct_lst = []
        with torch.no_grad():
            for batch_idx, minibatch in enumerate(test):  # full test data
                minibatch = set_device(minibatch, device)
                correct, total = model.evaluate(minibatch)
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
        correct, total = model.evaluate(minibatch)
        correct_lst.append(correct)
    print("last epoch, eval: " + str(((sum(correct_lst)/len(correct_lst)) / 32).item()))
'''
