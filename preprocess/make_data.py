import pandas as pd
import numpy as np



description = pd.read_pickle('../data/data.pkl')
audio = pd.read_pickle('../data/audio.pkl')

data = pd.merge(description, audio)
data = data.reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)

data3.to_pickle('../data/data.pkl')
