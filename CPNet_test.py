# -*- coding: utf-8 -*-

from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
# from davisinteractive.utils.visualization import overlay_mask
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import random
import sys


### My libs
sys.path.append('models/')
from DAVIS_dataset import DAVIS_Test
from models.CPNet_model import CPNet
 

def get_arguments():
    parser = argparse.ArgumentParser(description="CPNet")
    parser.add_argument("-b", type=int, default=1)
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-e", type=int, default=0)
    parser.add_argument("-D", type=str, default='/dataset') # dataset path
    return parser.parse_args()

args = get_arguments()
BATCH_SIZE = args.b
GPU = args.g
RESUME = args.e
DATA_ROOT = args.D


# Model and version
MODEL = 'CPNet_model'
# description
print(MODEL, ': with DAVIS final')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

print(torch.backends.cudnn.version())
print(torch.version.cuda)



model = nn.DataParallel(CPNet())
if torch.cuda.is_available():
    model.cuda()

Pset = DAVIS_Test(root=DATA_ROOT + '/DAVIS_example', imset='example.txt', size='half')
Trainloader = data.DataLoader(Pset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# resuming fine-tune
model.load_state_dict(torch.load(os.path.join('./weight/weight.pth')))

model.eval() # turn-off BN

num_length = 120

for i, V in tqdm.tqdm(enumerate(Trainloader)):
    frames, masks, GTs, info = V # b,3,t,h,w / b,1,t,h,w
    
    seq_name = info['name'][0]
    num_frames = frames.size()[2]
    print(seq_name, frames.size())

    with torch.no_grad():
        rfeats = model(frames, masks)
    frames_ = frames.clone()
    masks_ = masks.clone() 
    index = [f for f in reversed(range(num_frames))]
        
    for t in range(2): # forward : 0, backward : 1
        if t == 1:
            comp0 = frames.clone()
            frames = frames_
            masks = masks_
            index.reverse()

        for f in index:
            ridx = []
            
            start = f - num_length
            end = f + num_length

            if f - num_length < 0:
                end = (f + num_length) - (f - num_length)
                if end > num_frames:
                    end = num_frames -1
                start = 0

            elif f + num_length > num_frames:
                start = (f - num_length) - (f + num_length - num_frames)
                if start < 0:
                    start = 0
                end = num_frames -1
                
            # interval: 2
            for i in range(start, end,2):
                if i != f:
                    ridx.append(i)
            
            with torch.no_grad():
                comp = model(rfeats[:,:,ridx], frames[:,:,ridx], masks[:,:,ridx], frames[:,:,f], masks[:,:,f], GTs[:,:,f])
                
                c_s = comp.shape
                Fs = torch.empty((c_s[0], c_s[1], 1, c_s[2], c_s[3])).float().cuda()
                Hs = torch.zeros((c_s[0], 1, 1, c_s[2], c_s[3])).float().cuda()
                Fs[:,:,0] = comp.detach()
                frames[:,:,f] = Fs[:,:,0]
                masks[:,:,f] = Hs[:,:,0]                
                rfeats[:,:,f] = model(Fs, Hs)[:,:,0]

            save_path = os.path.join('./test/', seq_name)
            if t == 1:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                est = comp0[:,:,f] * (len(index)-f) / len(index) + comp.detach().cpu() *f / len(index)
                  
                canvas = (est[0].permute(1,2,0).numpy() * 255.).astype(np.uint8)
                
                if canvas.shape[1] % 2 != 0:
                    canvas = np.pad(canvas, [[0,0],[0,1],[0,0]], mode='constant')

                canvas = Image.fromarray(canvas)
                canvas.save(os.path.join(save_path, 'f{}.jpg'.format(f)))




    vid_path = os.path.join('./test/', '{}.mp4'.format(seq_name))
    frame_path = os.path.join('./test/', seq_name, 'f%d.jpg')
    os.system('ffmpeg -framerate 10 -i {} {}  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
    print('----------------------------------------------------------')


print('Done')
