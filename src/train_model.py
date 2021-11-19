# -*- coding: utf-8 -*-
# +
import sys

import torch
import pandas as pd
from torch.utils.data import DataLoader

from data import get_data_iterators
from data import set_device
from model import AudioTextModel
# -

ITER = 300

# +
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioTextModel(num_classes=10).to(device)

print("### load model")

# +
train, test = get_data_iterators()

print("### load data")

# +
print("### start train")
print("\n")

for step in range(ITER):   # epoch
        
    loss_lst = []
    for batch_idx, minibatch in enumerate(train):       # 한 번의 epoch                 
        minibatch = set_device(minibatch, device)
        loss = model.update(minibatch)
        loss_lst.append(loss)
        
    if (step+1) % 10 == 0:   # 10 epoch 마다 진행상황
        print('# step [{}/{}], loss: {}'.format(step + 1, ITER, (sum(loss_lst)/len(loss_lst))))
    
    if (step+1) % 30 == 0:   # 30 epoch 마다 evaluate
        correct_lst = []
        with torch.no_grad():
            for batch_idx, minibatch in enumerate(test):  # full test data
                minibatch = set_device(minibatch, device)
                correct, total = model.evaluate(minibatch)
                correct_lst.append(correct)
            print(str(step+1) + " epoch, eval: " + str(((sum(correct_lst)/len(correct_lst)) / 32).item()))

# last
correct_lst = []
with torch.no_grad():
    for batch_idx, minibatch in enumerate(test):  # full test data
        minibatch = set_device(minibatch, device)
        correct, total = model.evaluate(minibatch)
        correct_lst.append(correct)
    print("last epoch, eval: " + str((sum(correct_lst)/len(correct_lst)) / 32))
        
