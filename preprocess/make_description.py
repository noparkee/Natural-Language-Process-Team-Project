import pickle
import pandas as pd
import numpy as np
import os
import re

PARENT_PATH = 'data/IEMOCAP_full_release/Session'
DIALOG_PATH = '/dialog/transcriptions/'
WAV_PATH = '/sentences/wav/'
EMOTION_PATH = '/dialog/EmoEvaluation/'

wav_lst, sentence_lst = [], []
sentence_file_name = []

label_lst = []
v_lst, a_lst, d_lst = [], [], []
label_file_name = []

# [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
for i in range(1, 6):
    PATH_S = PARENT_PATH + str(i) + DIALOG_PATH
    PATH_E = PARENT_PATH + str(i) + EMOTION_PATH
    file_list = os.listdir(PATH_S)

    for f in file_list:
        SENTENCE_FILE = open(PATH_S+f, 'r')
        EMOTION_FILE = open(PATH_E+f, 'r')
        while True:
            line_s = SENTENCE_FILE.readline()
            line_e = EMOTION_FILE.readline()
            if not line_s:
                break
            
            s = re.findall("(.*) \[.*\]: (.*)", line_s)
            if len(s) != 0:           
                folder = re.findall("(S.*_.*)_(.*)", s[0][0])
                wav, sentence = (PARENT_PATH + str(i) + WAV_PATH + folder[0][0] + '/' + s[0][0] + '.wav'), s[0][1]
                wav_lst.append(wav)
                #print(wav)
                #input()
                sentence_lst.append(sentence)
                sentence_file_name.append(s[0][0])

            e = re.findall("\[.*\]\s(S.*)\s(.*)\s\[(.*), (.*), (.*)\]", line_e)
            if len(e) != 0:
                label, v, a, d = e[0][1], e[0][2], e[0][3], e[0][4]
                label_lst.append(label)
                v_lst.append(v)
                a_lst.append(a)
                d_lst.append(d)
                label_file_name.append(e[0][0])


        SENTENCE_FILE.close()
        EMOTION_FILE.close()

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

data = pd.read_pickle('../data/description.pkl')
list(set(data['label']))

list(data['label']).count('xxx')

list(data['label']).count('oth')

label_lst = ['xxx', 'ang', 'hap', 'exc', 'sad', 'fru', 'neu', 'sur', 'oth', 'fea']
for l in label_lst:
    print(list(data['label']).count(l))


def to_number(x):
    if x == 'xxx':
        return 0
    if x == 'ang':
        return 1
    if x == 'hap':
        return 2
    if x == 'exc':
        return 3
    if x == 'sad':
        return 4
    if x == 'fru':
        return 5
    if x == 'neu':
        return 6
    if x == 'sur':
        return 7
    if x == 'oth':
        return 8
    if x == 'fea':
        return 9


label_num = list(map(to_number, data['label']))
label_num

data['label_num'] = label_num

data.to_pickle('../data/description.pkl')


