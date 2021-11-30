import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import numpy as np



PATH = '../data/images/'

#import file description
data = pd.read_pickle('../data/description.pkl')
image_path_lst = []

for i in range(len(data)):
    file_path = data['wav_path'][i]
    name = data['name'][i]

    #y, sr = librosa.load(file_path, sr=44100)

    #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, fmax=sr/2)
    #S_dB = librosa.power_to_db(S, ref=np.max)

    #img = librosa.display.specshow(S_dB)
    
    img_path = PATH + name + '.png'
    #plt.savefig(img_path)
    image_path_lst.append(img_path)

    #plt.close()

data['image_path'] = image_path_lst
data.to_pickle('../data/description.pkl')
