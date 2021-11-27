import pickle
import pandas as pd
import numpy as np
import os
import re

PARENT_PATH = '../data/IEMOCAP_full_release/Session'
DIALOG_PATH = '/dialog/transcriptions/'
WAV_PATH = '/sentences/wav/'
EMOTION_PATH = '/dialog/EmoEvaluation/'

wav_lst, sentence_lst = [], []
sentence_file_name = []

label_lst = []
v_lst, a_lst, d_lst = [], [], []
label_file_name = []

# +
# [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]

# Session1 ~ Session5
for i in range(1, 6):
    
    PATH_S = PARENT_PATH + str(i) + DIALOG_PATH     # Session1/dialog/transcriptions (contains Ses01F_impro01.txt)
    PATH_E = PARENT_PATH + str(i) + EMOTION_PATH    # Session1/dialog/EmoEvaluation (contains Ses01F_impro01.txt)
    file_list = os.listdir(PATH_S)   # Ses01F_impro01.txt

    
    for f in file_list:
        SENTENCE_FILE = open(PATH_S+f, 'r')
        EMOTION_FILE = open(PATH_E+f, 'r')
        while True:
            line_s = SENTENCE_FILE.readline()
            if not line_s:
                break
            
            s = re.findall("(.*) \[.*\]: (.*)", line_s)
            if len(s) != 0:           
                folder = re.findall("(S.*_.*)_(.*)", s[0][0])
                wav, sentence = (PARENT_PATH + str(i) + WAV_PATH + folder[0][0] + '/' + s[0][0] + '.wav'), s[0][1]
                wav_lst.append(wav)
                #print(s[0][0])
                #input()
                sentence_lst.append(sentence)
                sentence_file_name.append(s[0][0])

        while True:
            line_e = EMOTION_FILE.readline()
            if not line_e:
                break

            e = re.findall("\[.*\]\s(S.*)\s(.*)\s\[(.*), (.*), (.*)\]", line_e)
            if len(e) != 0:
                label, v, a, d = e[0][1], e[0][2], e[0][3], e[0][4]
                label_lst.append(label)
                v_lst.append(v)
                a_lst.append(a)
                d_lst.append(d)
                label_file_name.append(e[0][0])
                #print(e[0][0]+" "+label)
                #print()



        SENTENCE_FILE.close()
        EMOTION_FILE.close()

# +
#10087
len(wav_lst)
len(sentence_lst)
len(sentence_file_name)
#10039
len(label_lst)
len(label_file_name)

len(wav_lst), len(d_lst)
#label_file_name

## 3: (2158, 2136)
## 5: (2196, 2170)
# -

sentence_pd = pd.DataFrame({'name': sentence_file_name, 'wav_path': wav_lst, 'sentence': sentence_lst})
label_pd = pd.DataFrame({'name': label_file_name, 'label': label_lst, 'v': v_lst, 'a': a_lst, 'd': d_lst})

data = pd.merge(sentence_pd, label_pd)
data = data.sort_values(by='name')
data.to_pickle('../data/description.pkl')

# tmp = pd.read_pickle('description.pkl')
# tmp = tmp.sort_values(by='name')
# print(tmp)

'''print(wav_lst[:5])
print(sentence_lst[:5])
print(label_lst[:5])

print(sentence_file_name[:5])
print(label_file_name[:5])'''

data = pd.read_pickle("../data/description.pkl")
list(set(data["label"]))

list(set(data["label_num"]))

list(data['label']).count('xxx')

list(data['label']).count('oth')

label_lst = ['fru', 'neu', 'hap', 'sad', 'sur', 'dis', 'xxx', 'exc', 'oth', 'fea', 'ang']
for l in label_lst:
    print(l + ' ' + str(list(data['label']).count(l)))


def to_number(x):
    if x == 'fru':
        return 0
    if x == 'neu':
        return 1
    if x == 'hap':
        return 2
    if x == 'sad':
        return 3
    if x == 'sur':
        return 4
    if x == 'dis':
        return 5
    if x == 'xxx':
        return 6
    if x == 'exc':
        return 7
    if x == 'oth':
        return 8
    if x == 'fea':
        return 9
    if x == 'ang':
        return 10


label_num = list(map(to_number, data['label']))
label_num

data['label_num'] = label_num

# +
# --- #
# -

data = pd.read_pickle("../data/description.pkl")

lst = ['xxx', 'dis', 'oth', 'fea']
for l in lst:
    data = data.drop(data[data['label'] == l].index)

data.loc[data['label'] == 'sur', 'label'] = 'exc'

list(set(data["label"]))


def to_number(x):
    if x == 'fru':
        return 0
    if x == 'neu':
        return 1
    if x == 'hap':
        return 2
    if x == 'sad':
        return 3
    if x == 'exc':
        return 4
    if x == 'ang':
        return 5


label_num = list(map(to_number, data['label']))
data['label_num'] = label_num

data

data = data.reset_index(drop=True)

data.to_pickle('../data/description2.pkl')

data = pd.read_pickle('../data/description2.pkl')

for l in list(set(data["label"])):
    print(l + ' ' + str(list(data['label']).count(l)))

data

data = data.sample(frac=1).reset_index(drop=True)
data

data.to_pickle('../data/description3.pkl')

pd.read_pickle('../data/description3.pkl')

data = pd.read_pickle('../data/description2.pkl')
list(set(data['label']))

data.loc[data['label'] == 'hap', 'label'] = 'exc'

list(set(data['label']))


def to_number(x):
    if x == 'neu':
        return 0
    if x == 'exc':
        return 1
    if x == 'sad':
        return 2
    if x == 'fru':
        return 3
    if x == 'ang':
        return 4


label_num = list(map(to_number, data['label']))
data['label_num'] = label_num

data.to_pickle('../data/description4.pkl')

lst = list(set(data['label']))
for l in lst:
    print(l, list(data['label']).count(l))


