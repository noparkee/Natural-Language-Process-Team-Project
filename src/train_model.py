# -*- coding: utf-8 -*-
# +
import os
import time
import sys
import random
import numpy as np
import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader

from data import get_data_iterators, set_device
from model import AudioTextModel
from utils import set_seed, get_score
# -

## Hyperparameters
ITER = 4
NUM_CLASSES = 4
BATCH_SIZE = 32

## global seed 고정
set_seed(14)



# ---
#print("### pid: ", os.getpid())
affinity_mask = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
os.sched_setaffinity(0, affinity_mask)
# ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('### device: ' + str(device))
model = AudioTextModel(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES).to(device)

### Check the learnable parameters
# for name, param in model.named_parameters():
#     if(param.requires_grad):
#         print(name)
#     else:
#         print('no grad',name)

print("### load model")

train, test = get_data_iterators(BATCH_SIZE)
print("### load data")

print("### start train")
print("\n")
for step in range(ITER):   # epoch
    model.train()

    model.debuglst_train = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    model.debuglst_test = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    ### training
    train_correct, train_loss, train_num_batch = 0, 0, 0
    train_starttime = time.time()

    for batch_idx, minibatch in enumerate(train):       # 한 번의 epoch    
        minibatch = set_device(minibatch, device)
        correct, loss = model.update(minibatch)
        #text_correct, audio_correct, correct, text_loss, audio_loss, loss = model.update(minibatch)
    
        train_correct += correct
        train_loss += loss
        train_num_batch += 1

    train_time = time.time()
    ua, wa = get_score(model.debuglst_train, NUM_CLASSES)
    print('# step [{}/{}], train loss: {}'.format(step + 1, ITER, (train_loss / train_num_batch)))
    #print('# step [{}/{}], train unweighted accuracy (UA): {}'.format(step + 1, ITER, (train_correct / train_num_batch) / BATCH_SIZE))
    print('# step [{}/{}], train unweighted accuracy (UA): {}'.format(step + 1, ITER, ua))
    print('# step [{}/{}], train weighted accuracy (WA): {}'.format(step + 1, ITER, wa))
    train_correct, train_loss, train_num_batch = 0, 0, 0

    ### testing (validation)
    model.eval()

    test_correct, test_loss, test_num_batch = 0, 0, 0
    test_starttime = time.time()
    for batch_idx, minibatch in enumerate(test):
        minibatch = set_device(minibatch, device)
        with torch.no_grad():
            correct, loss = model.evaluate(minibatch)
            #text_correct, audio_correct, correct, text_loss, audio_loss, loss = model.evaluate(minibatch)
        
        test_correct += correct
        test_loss += loss
        test_num_batch += 1

    test_time = time.time()
    ua, wa = get_score(model.debuglst_test, NUM_CLASSES)
    print('# step [{}/{}], test loss: {}'.format(step + 1, ITER, (test_loss / test_num_batch)))
    #print('# step [{}/{}], test unweighted accuracy (UA): {}'.format(step + 1, ITER, (test_correct / test_num_batch) / BATCH_SIZE))
    print('# step [{}/{}], test unweighted accuracy (UA): {}'.format(step + 1, ITER, ua))
    print('# step [{}/{}], test weighted accuracy (WA): {}'.format(step + 1, ITER, wa))
    test_correct, test_loss, test_num_batch = 0, 0, 0

    print('----------')
    print("training time: ", (train_time - train_starttime), "sec")
    print("testing time: ", (test_time - test_starttime), "sec")

    print(model.debuglst_train)
    print(model.debuglst_test)

    print("==========\n")
