#! /media/md01/home/ne6111089/anaconda3/envs/taco/bin/python3.9

import os
import glob
import wave
import scipy
import numpy as np

def wav_split(wav,emotions):
    (nchannels, sampwidth, framerate, nframes, comptype, compname),samples = wav
    left = samples[0::nchannels]
    right = samples[1::nchannels]

    frames = []
    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']

        e['right'] = right[int(start * framerate):int(end * framerate)]
        e['left'] = left[int(start * framerate):int(end * framerate)]

        frames.append({'left': e['left'], 'right': e['right']})
    return frames


def padding_sequence(xt,max_len=None,truncating='post',padding='post',value=0.):
    Nsamples = len(xt)
    if max_len is None:
        lengths = [t.shape(0) for t in xt] #all sequences are np.array and put in the list
        max_len = np.max(lengths)
    
    x_out = np.ones(shape=[Nsamples,max_len]+list(xt[0].shape[1:]),dtype=xt[0].dtype)* np.asarray(value, dtype=xt[0].dtype)
    mask = np.zeros(shape=[Nsamples,max_len],dtype=x_out.dtype)
    
    for i in range(Nsamples):
        x = xt[i]
        if truncating == 'pre':
            trunc = x[-max_len:]
        elif truncating == 'post':
            trunc = x[:max_len]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            x_out[i, :len(trunc)] = trunc
            mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            x_out[i, -len(trunc):] = trunc
            mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x_out, mask


def get_audio(wav_path,filename):
    wav = wave.open(wav_path + filename, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    samples = np.fromstring(content, dtype=np.int16)
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples


def get_transcription(trans_path,filename):
    f = open(trans_path + filename,mode='r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    ##split the dialog in to speaker_id --> sentence
    for i in range(len(f)-1):
        sentence = f[i]
        
        id_end = sentence.find(' [')
        sen_start = sentence.find(': ')
        
        id = sentence[:id_end]
        sen = sentence[sen_start+2:] # 2 because of the two character ':' and ' '
        
        transcription[id] = sen
        
    return transcription
   
   
def get_label(emotion_path,filename):
    f = open(emotion_path+filename,mode='r').read()
    f = np.array(f.split('\n'))
    ##Record the empty line with True .
    idx = f == ''
    ##Use the np.arange() to record the empty line
    idx_n = np.arange(len(f))[idx]
    
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i]+1:idx_n[i+1]]
        #With the summary condition : [start,end] speaker emotion [V,A,D]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                        head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])
        
        j = 1
        emos = []
        while g[j][0] == "C":
            head = g[j]
            start_idx = head.find("\t") + 1
            evoluator_emo = []
            idx = head.find(";", start_idx)
            while idx != -1:
                evoluator_emo.append(head[start_idx:idx].strip().lower()[:3])
                start_idx = idx + 1
                idx = head.find(";", start_idx)
            emos.append(evoluator_emo)
            j += 1

        emotion.append({'start': start_time,
                        'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo,
                        'emo_evo': emos})
    return emotion
    

def get_field(data, key):
    return np.array([e[key] for e in data])

def convert_gt_from_array_to_list(gt_batch, gt_batch_mask=None):

    B, L = gt_batch.shape
    gt_batch = gt_batch.astype('int')
    gts = []
    for i in range(B):
        if gt_batch_mask is None:
            l = L
        else:
            l = int(gt_batch_mask[i, :].sum())
        gts.append(gt_batch[i, :l].tolist())
    return gts
     