#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9

import os
import pickle
from data import *


frame_rate = 16000
emotion_class = ['neu','ang','sad','exc']
data_path = "/media/hd03/ne6111089_data/IEMOCAP/"
sessions = ['Session1','Session2','Session3','Session4','Session5']

def read_data_emo():
    data = []
    ids = {}
    for session in sessions:
        wav_files = data_path + session + '/dialog/wav/'
        emotion_files = data_path + session + '/dialog/EmoEvaluation/'
        transcript_files = data_path +session +'/dialog/transcriptions/'
        
        files2 = os.listdir(wav_files)
        files = []
        
        for i in files2:
            if(i.endswith(".wav")):
                if i[0]=='.':
                    files.append(i[2:-4])
                else:
                    files.append(i[:-4])

        for f in files:       
            print(f)
            mocap_f = f
            if (f== 'Ses05M_script01_1b'):
                mocap_f = 'Ses05M_script01_1' 
            
            wav = get_audio(wav_files, f + '.wav')
            transcriptions = get_transcription(transcript_files, f + '.txt')
            emotions = get_label(emotion_files, f + '.txt')
            sample = wav_split(wav, emotions)
            
            
            ## emotions is the dictionary with id and emotion
            for ie, e in enumerate(emotions):
                '''if 'F' in e['id']:
                    e['signal'] = sample[ie]['left']
                else:
                    e['signal'] = sample[ie]['right']'''
                
                e['signal'] = sample[ie]['left']
                e.pop("left", None)
                e.pop("right", None)
                e['transcription'] = transcriptions[e['id']]
                
                if e['emotion'] in emotion_class:
                    if e['id'] not in ids:
                        data.append(e)
                        ids[e['id']] = 1

                        
    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]
            

data = read_data_emo()
#data_path = "/media/md01/home/ne6111089/Tacotron2-PyTorch/ser/"
print(len(data))
with open('data.pickle','wb') as handle:
    pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)
