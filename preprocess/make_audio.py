# -*- coding: utf-8 -*-
import pandas as pd
import librosa
import pickle

#import file description
data = pd.read_pickle('../data/description.pkl')
data = data[['name','wav_path']]
data = data.sort_values(by='name')
data

# +
##try one data
file_path = data['wav_path'][3]

##default 44100
y, sr = librosa.load(file_path, sr=44100)
y, sr
librosa.feature.mfcc(y,).shape
# -

##run all
data['wav_vec'] = data['wav_path'].apply(lambda x: librosa.load(x,sr=3840)[0])

data['mfcc'] = data['wav_vec'].apply(lambda x: librosa.feature.mfcc(x))

data.to_pickle('data/audio_vec.pkl')

# +
data = pd.read_pickle('../data/audio_vec.pkl')

#data['mfcc'][3].shape
data

# +
import numpy as np

### test
tmp = list(data['mfcc'].map(lambda x: 20*(x.shape[1])))
print((sum(tmp)/len(tmp)))
print(max(tmp))

### 진짜 최대가 4020인지 확인
tmp = list(data['mfcc'].map(lambda x: np.array(x).flatten().tolist()))

max_len = 0
for i in tmp:
        if max_len < len(i):
            max_len = len(i)

print(max_len)
# -


