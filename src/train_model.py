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
ITER = 10
NUM_CLASSES = 5
BATCH_SIZE = 32

## global seed 고정
SEED = 14
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
#print("### pid: ", os.getpid())
affinity_mask = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
os.sched_setaffinity(0, affinity_mask)
# ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('### device: ' + str(device))
model = AudioTextModel(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES).to(device)
print("### load model")

train, test = get_data_iterators(BATCH_SIZE)
print("### load data")

print("### start train")
print("\n")
for step in range(ITER):   # epoch
        
    model.debuglst_test = torch.zeros(NUM_CLASSES, NUM_CLASSES)
    model.debuglst_train = torch.zeros(NUM_CLASSES, NUM_CLASSES)

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
    
    print('# step [{}/{}], train loss: {}'.format(step + 1, ITER, (train_loss / train_num_batch) / BATCH_SIZE))
    print('# step [{}/{}], train accuracy: {}'.format(step + 1, ITER, (train_correct / train_num_batch) / BATCH_SIZE))
    train_correct, train_loss, train_num_batch = 0, 0, 0


    ### testing (validation)
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
    
    print('# step [{}/{}], test loss: {}'.format(step + 1, ITER, (test_loss / test_num_batch) / BATCH_SIZE))
    print('# step [{}/{}], test accuracy: {}'.format(step + 1, ITER, (test_correct / test_num_batch) / BATCH_SIZE))
    test_correct, test_loss, test_num_batch = 0, 0, 0

    print('----------')
    print("training time: ", (train_time - train_starttime), "sec")
    print("testing time: ", (test_time - test_starttime), "sec")

    print(model.debuglst_train)
    print(model.debuglst_test)

    print("==========\n")
