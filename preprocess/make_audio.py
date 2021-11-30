import pandas as pd
import librosa
import pickle
import numpy as np
import torch



# import file description
audio = pd.read_pickle('../data/description.pkl')
audio = audio[['name','wav_path']]
audio = audio.sort_values(by='name')

# run all
audio['wav_vec'] = audio['wav_path'].apply(lambda x: librosa.load(x,sr=44100)[0])   # 3840   # 16800
audio['mfcc'] = audio['wav_vec'].apply(lambda x: librosa.feature.mfcc(x, sr=44100))

audio['mfcc_tensor'] = audio['mfcc'].map(lambda x: torch.tensor(x))
audio['len'] = audio['mfcc'].map(lambda x: x.shape[1])

audio = audio.reset_index(drop=True)
audio = audio.sample(frac=1).reset_index(drop=True)

audio.to_pickle('../data/audio_vec.pkl')
