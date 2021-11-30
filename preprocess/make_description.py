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


sentence_pd = pd.DataFrame({'name': sentence_file_name, 'wav_path': wav_lst, 'sentence': sentence_lst})
label_pd = pd.DataFrame({'name': label_file_name, 'label': label_lst, 'v': v_lst, 'a': a_lst, 'd': d_lst})

description = pd.merge(sentence_pd, label_pd)
description = description.sort_values(by='name')
#description.to_pickle('../data/description.pkl')

#description = pd.read_pickle("../data/description4.pkl")


label_lst = list(set(description['label']))
for l in label_lst:
    print(l + ' ' + str(list(description['label']).count(l)))

# delete label
lst = ['fru', 'sur', 'dis', 'xxx', 'oth', 'fea']
for l in lst:
    description = description.drop(description[description['label'] == l].index)

# change exc to hap
description.loc[description['label'] == 'exc', 'label'] = 'hap'


def to_number(x):
    if x == 'neu':
        return 0
    if x == 'hap':
        return 1
    if x == 'sad':
        return 2
    if x == 'ang':
        return 3

label_num = list(map(to_number, description['label']))
description['label_num'] = label_num

print("===")

label_lst = list(set(description['label']))
for l in label_lst:
    print(l + ' ' + str(list(description['label']).count(l)))

description = description.reset_index(drop=True)
description = description.sample(frac=1).reset_index(drop=True)

description.to_pickle('../data/description.pkl')
