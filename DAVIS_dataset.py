import matplotlib
matplotlib.use('Agg')
import os
import os.path as osp
import numpy as np
from PIL import Image

import collections
import torch
import torchvision
from torch.utils import data

import glob
import time
import cv2
import random
import csv

import tqdm

import matplotlib.pyplot as plt


class DAVIS_Test(data.Dataset):
    def __init__(self, root, imset='2016/trainval.txt', resolution='480p', size=None):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.targets = {}
        self.shape = {}
        self.video_target = []
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                self.targets[_video] = [[x] for x in range(1,self.num_objects[_video]+1)] # + self.grp[_video] 
                for t in self.targets[_video]:
                    self.video_target.append({'video':_video, 'target':t})
        self.size = size

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        info = {}
        rnd = random.Random()
        vt = self.video_target[index]
        video = self.videos[index]
        info['name'] = video

        N = self.num_frames[video]
        idxs = range(N)
            

        H, W = self.shape[video]
        if self.size:
            if self.size == 'half':
                # H, W = int(H / 2.), int(W / 2.)
                H, W = 240, 424
            
            elif len(self.size) == 2:
                H, W = self.size
            else:
                pass

        # make sure dividable by 8
        d = 8
        if H % d > 0:
            if H % d > d/2:
                new_H = H + d - H % d
            else:
                new_H = H - H % d
        else:
            new_H = H

        if W % d > 0:
            if W % d > d/2:
                new_W = W + d - W % d
            else:
                new_W = W - W % d
        else:
            new_W = W

        H, W = new_H, new_W

        N_frames = np.empty((N, H, W, 3), dtype=np.float32)
        N_masks = np.empty((N, H, W, 1), dtype=np.float32)

        for i, f in enumerate(idxs):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            raw_frame = np.array(Image.open(img_file).convert('RGB'))/255.
            N_frames[i] = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_LINEAR)

            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
            raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            raw_mask = (raw_mask > 0.5).astype(np.uint8)
            raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            N_masks[i,:,:,0] = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=4).astype(np.float32)

        Fs = torch.from_numpy(np.transpose(N_frames, (3, 0, 1, 2)).copy()).float()
        Hs = torch.from_numpy(np.transpose(N_masks, (3, 0, 1, 2)).copy()).float()
      
        GTs = Fs

        Fs = (1-Hs)*GTs + Hs*torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1,1)

        return Fs, Hs, GTs, info
    