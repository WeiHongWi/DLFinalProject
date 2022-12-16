#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9

import torch
import torch.nn as nn
from model import SERmodel_speech_text
from loader import train_st_dl,test_st_dl


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model = SERmodel_speech_text()
#model= nn.DataParallel(model,device_ids = [1,2])
model = model.to(device)
criteria = nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(),lr=0.0002)
epochs = 20
batch_size = 4
def train(model,epochs):
    model.train()
    for i in range(epochs):
        epoch_loss = 0.0
        for audio,text,label in train_st_dl:
            audio = audio.to(device)
            text = text.to(device)
            label = label.to(device)
            label = label.squeeze(1)
            label = label.long()
            
            optimizer.zero_grad()
            pred,_ = model(audio,text,mode='train',device=device)
            loss = criteria(pred,label)
            epoch_loss += loss.detach()
            
            loss.backward()
            optimizer.step()
            
        print(f'{i}/{epochs}: '+f'The loss of the model --> {epoch_loss/505}')
        torch.save(model.state_dict(), 'checkpoints.pt')

if __name__ == '__main__':
    train(model,epochs)