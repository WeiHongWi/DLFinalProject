#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9

import torch
import pickle
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

emotion_class ={'neu':0.0,'ang':1.0,'sad':2.0,'exc':3.0}
    

speech = np.load('speech.npy')
train_speech ,test_speech= train_test_split(speech,random_state=777,train_size=0.9)
#print(train_speech.shape)
train_speech = train_speech[:,np.newaxis,:,:]
test_speech = test_speech[:,np.newaxis,:,:]

with open("text.pickle",'rb') as file:
    text_list = pickle.load(file)

train_text,test_text = train_test_split(text_list,random_state=777,train_size=0.9)


train_max_len = 0
test_max_len = 0
for i in range(len(train_text)):
    train_text[i] = train_text[i].squeeze(0)
    length = int(train_text[i].shape[0])
    if length>train_max_len:
        train_max_len = length
    
for j in range(len(test_text)):
    test_text[j] = test_text[j].squeeze(0)
    #length1 = test_text[i].shape[0]
    #if length1>test_max_len:
        #test_max_len = length1

print(train_max_len)
print(test_max_len)

label = np.load('label.npy')
train_label,test_label = train_test_split(label,random_state=777,train_size=0.9)

transform = transforms.Compose([transforms.ToTensor()])

train_speech = torch.from_numpy(train_speech)
test_speech = torch.from_numpy(test_speech)




train_label_fix = []
for i in range(len(train_label)):
    train_label_fix.append(emotion_class[train_label[i]])

    
train_label_fix = np.array(train_label_fix)
train_label_fix = train_label_fix[:,np.newaxis]
train_label_fix = transform(train_label_fix)
train_label_fix = train_label_fix.squeeze(0)

test_label_fix = []
for i in range(len(test_label)):
    test_label_fix.append(emotion_class[test_label[i]])

    
test_label_fix = np.array(test_label_fix)
test_label_fix = test_label_fix[:,np.newaxis]
test_label_fix = transform(test_label_fix)
test_label_fix = test_label_fix.squeeze(0)


class speech_data(Dataset):
    def __init__(self,speech,label):
        self.speech = speech
        self.label = label
    
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self,index):
        audio = self.speech[index]
        label = self.label[index]
        
        return audio,label
    
class speech_text_data(Dataset):
    def __init__(self,speech,text,label):
        self.speech = speech
        self.text = text
        self.label = label
    
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self,index):
        audio = self.speech[index]
        label = self.label[index]
        text = self.text[index]
        
        return audio,text,label

train_dataset = speech_text_data(train_speech,train_text,train_label_fix)
test_dataset = speech_text_data(test_speech,test_text,test_label_fix)

def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    return data

train_st_dl = DataLoader(train_dataset,shuffle=True,batch_size=16)
test_st_dl = DataLoader(test_dataset,shuffle=True,batch_size=16)

for audio,text,label in train_st_dl:
    print(text)
    break
