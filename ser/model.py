#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence


class SERmodel_speech(nn.Module):
    def __init__(self,audio_feature_row,audio_feature_col):
        '''About the speech data 
        (1) Flatten
        (2) ( Dense layer + Relu + Dropout ) * 3
        '''
        self.flatten = nn.Flatten((audio_feature_row,audio_feature_col))
        self.linear = nn.Linear(audio_feature_row*audio_feature_col,1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(0.2)
        
        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,256)
        
        self.Dense_classifier = nn.Linear(256,4)

        
    def forward(self,audio):

            
        output = self.flatten(audio)
        output = self.linear(output)
        output = self.relu(output)
        output =self.dropout(output)
        output = self.linear1(output)
        output = self.relu(output)
        output =self.dropout(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.Dense_classifier(output)
   
        return output
        
        
class SERmodel_speech_text(nn.Module):
    def __init__(self):
        '''
        About the speech data :(1,39,998)
        (1) Flatten
        (2) ( Dense layer + Relu + Dropout ) * 3
        
        About the text data : (1,384)
        (1) (Conv1D + Relu + Dropout) * 3
        
        '''
        super(SERmodel_speech_text,self).__init__()
        self.flatten = nn.Flatten()
        self.conv2d = nn.Conv2d(1,8,(9,26))
        self.pool = nn.MaxPool2d(3)
        self.dropout = nn.Dropout(0.2)
        
        self.conv2d1 = nn.Conv2d(8,16,kernel_size=3)
        self.pool1 = nn.MaxPool2d(3)
        self.dropout1 = nn.Dropout(0.2)
        
        #self.conv2d2 = nn.Conv2d(32,128,kernel_size=3)
        
       
        self.lstm = nn.LSTM(input_size=3072,hidden_size=1024,batch_first=True)
        self.lstm1 = nn.LSTM(input_size=1024,hidden_size=512,batch_first=True)
        self.linear = nn.Linear(512,256)
        
    
        
        
    def forward(self,audio,text,mode,device):
        if mode == 'train':
            output1 = self.conv2d(audio)
            rett = output1
            output1 = self.pool(output1)
            output1 = self.dropout(output1)
        
            output1 = self.conv2d1(output1)
            output1 = self.pool1(output1)
            output1 = self.dropout1(output1)
        
            #output1 = self.conv2d2(output1)
            output1 = self.flatten(output1)
            print(output1.shape[1])
            flatten_speech = nn.Linear(output1.shape[1],4096)
            flatten_speech.to(device)
            flatten_speech_2 = nn.Linear(4096,256)
            flatten_speech_2.to(device)
            output1 = flatten_speech(output1)
            output1 = flatten_speech_2(output1)

            packed_text = torch.nn.utils.rnn.pack_padded_sequence(text,batch_first=True,)
            output2 = self.lstm(text)

            audio_text = torch.cat((output1,output2),dim=1)
            output = self.classifier(audio_text)
   
            return output,rett        
        elif mode == 'calculate the loss':
            output1 = self.conv2d(audio)
            return output1
            
            
def collate_fn(data):
    data = data.sort(lambda x:len(x),reverse=True) 
    data = pad_sequence(data,batch_first=True,padding_value=0)
    return data