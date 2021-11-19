# -*- coding: utf-8 -*-
# +
import torch
import pandas as pd
from torch.utils.data import DataLoader

from data import get_data_iterators
from data import set_device
from model import AudioTextModel
# -

ITER = 1000

# +
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioTextModel(num_classes=10).to(device)

print("### load model ###")

# +
train, test = get_data_iterators()

print("### load data ###")

# +
print("### start train ###")

for i in range(ITER):   # epoch
    for batch_idx, minibatch in enumerate(train):  
        if i % 300 == 0 and batch_idx == 0:  # epoch-1 에서 test
            correct_lst = []
            with torch.no_grad():
                for batch_idx, minibatch in enumerate(test):  # full test data
                    minibatch = set_device(minibatch, device)
                    correct, total = model.evaluate(minibatch)
                    correct_lst.append(correct)
                print(str(i) + " epoch, eval: " + str((sum(correct_lst)/len(correct_lst)) / 32))
                      
        minibatch = set_device(minibatch, device)
        loss_dict = model.update(minibatch)

# last
correct_lst = []
with torch.no_grad():
    for batch_idx, minibatch in enumerate(test):  # full test data
        minibatch = set_device(minibatch, device)
        correct, total = model.evaluate(minibatch)
        correct_lst.append(correct)
    print("last epoch, eval: " + str((sum(correct_lst)/len(correct_lst)) / 32))
        
