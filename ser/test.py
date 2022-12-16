#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9
import torch
import torch.nn as nn
from loader import train_st_dl,test_st_dl
from model import SERmodel_speech_text

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = SERmodel_speech_text()
model.load_state_dict(torch.load('checkpoints.pt'))
model.eval()
model.to(device)

softmax = nn.Softmax()
with torch.no_grad():
    for audio,text,label in test_st_dl:
        audio = audio.to(device)
        text = text.to(device)
        label =label.to(device)
        label = label.squeeze(1)
        label = label.long()
        output,ret = model(audio,text,mode='train',device=device)
        print(label)
        print(output.shape)
        pred = softmax(output)
        print(pred)
        break
