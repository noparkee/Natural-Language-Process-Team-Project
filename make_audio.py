import pandas as pd
import librosa
import pickle

#import file description
data = pd.read_pickle('data/description.pkl')
data = data[['name','wav_path']]
data = data.sort_values(by='name')
data

# +
##try one data
file_path = data['wav_path'][3]

##default 44100
y, sr = librosa.load(file_path, sr=15000)
y, sr
librosa.feature.mfcc(y,).shape
# -

##run all
data['wav_vec'] = data['wav_path'].apply(lambda x: librosa.load(x,sr=44100)[0])

data['mfcc'] = data['wav_vec'].apply(lambda x: librosa.feature.mfcc(x))

data.to_pickle('data/audio_vec.pkl')

# +
data = pd.read_pickle('data/audio_vec.pkl')

data['mfcc'][7].shape
data
