#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9
import pickle
import librosa
import numpy as np


with open('data.pickle','rb') as handle:
    data_dict = pickle.load(handle)
    
x_train_speech = []
x_train_label = []


sample_rate = 16000
max_len = 1067  #Use the for loop to calculate it.

for one_example in data_dict:
    wav_sample = one_example['signal']
    wav_sample = np.float32(wav_sample)
    mel = librosa.feature.melspectrogram(wav_sample,sr=sample_rate)
    mel = librosa.util.normalize(mel)


    #if mel.shape[1]>=max_len:
        #max_len = mel.shape[1]
    padding_audio = np.zeros((128,max_len-mel.shape[1]),dtype='float32')
    mel = np.concatenate((mel,padding_audio),axis=1)
    

    x_train_speech.append(mel)
    
#print(max_len)
x_train_speech = np.array(x_train_speech)

#print(len(x_train_speech))



for example in data_dict:
    x_train_label.append(example['emotion'])
    
#print(len(x_train_label))



np.save('speech',x_train_speech)
np.save('label',x_train_label)
    
    
    
    
     