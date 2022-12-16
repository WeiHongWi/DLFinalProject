#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9

import torch
import pickle
import numpy as np
from pytorch_pretrained_bert import BertTokenizer,BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('/media/hd03/ne6111089_data/bertbaseuncased')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_bert.eval()
with open('data.pickle','rb') as handle:
    data_dict = pickle.load(handle)
    
x_train_text = []


for data in data_dict:
    text = data['transcription']
    marked_text = "[CLS] "+text+" [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #for tup in zip(tokenized_text, indexed_tokens):
        #print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    #break
    
    token_tensor = torch.tensor([indexed_tokens])
   
    with torch.no_grad():

        output = model_bert(token_tensor)
        
        all_hidden_states = output[0]
        word_embeddings = torch.cat([output[0][i] for i in [-1,-2,-3,-4]],dim=-1)
        x_train_text.append(word_embeddings)

with open("text.pickle",'wb') as handles:
    pickle.dump(x_train_text,handles)