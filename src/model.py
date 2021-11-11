import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.networks import AudioFeaturizer, BertEmbed, EmotionClassifier

class AudioTextModel(torch.nn.Module):
    def __init(self, num_classes):
        super(AudioTextModel, self).__init__()
        self.audioEmbedding = AudioFeaturizer(???)
        self.textEmbedding = BertEmbed()
        
        self.audio_projection = nn.Linear(???, 512)
        self.text_projection = nn.Linear(768, 512)
        self.classifier = EmotionClassifier(num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        

