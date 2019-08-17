from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
from torchvision import transforms

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
import sys

from .model_module import *
sys.path.insert(0, '.')
# from .common import *
sys.path.insert(0, '../utils/')
 
print('CPNet: Model')


# Alignment Encoder
class A_Encoder(nn.Module):
    def __init__(self):
        super(A_Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU()) # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 4
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv34 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 8
        self.conv4a = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 8
        self.conv4b = Conv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 8
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = F.upsample(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        x = self.conv34(x)
        x = self.conv4a(x)
        x = self.conv4b(x)
        return x

# Alignment Regressor
class A_Regressor(nn.Module):
    def __init__(self):
        super(A_Regressor, self).__init__()
        self.conv45 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 16
        self.conv5a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 16
        self.conv5b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 16
        self.conv56 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 32
        self.conv6a = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 32
        self.conv6b = Conv2d(512, 512, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 32
        init_He(self)
        
        self.fc = nn.Linear(512, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.conv45(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv56(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(-1, x.shape[1])

        theta = self.fc(x)
        theta = theta.view(-1, 2, 3)

        return theta

# Encoder (Copy network)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv12 = Conv2d(4, 64, kernel_size=5, stride=2, padding=2, activation=nn.ReLU()) # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        self.conv23 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.ReLU()) # 4
        self.conv3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.value3 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=None) # 4
        init_He(self)
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
    def forward(self, in_f, in_v):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v], dim=1)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3(x)
        v = self.value3(x)
        return v

# Decoder (Paste network)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        self.conv5_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU())
        
        # dilated convolution blocks
        self.convA4_1 = Conv2d(257, 257, kernel_size=3, stride=1, padding=2, D=2, activation=nn.ReLU())
        self.convA4_2 = Conv2d(257, 257, kernel_size=3, stride=1, padding=4, D=4, activation=nn.ReLU())
        self.convA4_3 = Conv2d(257, 257, kernel_size=3, stride=1, padding=8, D=8, activation=nn.ReLU())
        self.convA4_4 = Conv2d(257, 257, kernel_size=3, stride=1, padding=16, D=16,activation=nn.ReLU())

        self.conv3c = Conv2d(257, 257, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv3b = Conv2d(257, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv3a = Conv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 4
        self.conv32 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.ReLU()) # 2
        self.conv21 = Conv2d(64, 3, kernel_size=5, stride=1, padding=2, activation=None) # 1

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    
    def forward(self, x):
        x = self.conv4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.convA4_1(x)
        x = self.convA4_2(x)
        x = self.convA4_3(x)
        x = self.convA4_4(x)

        x = self.conv3c(x)
        x = self.conv3b(x)
        x = self.conv3a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv21(x)

        p = (x *self.std) + self.mean
        return p


# Context Matching Module
class CM_Module(nn.Module):
    def __init__(self):
        super(CM_Module, self).__init__()
    
    def masked_softmax(self, vec, mask, dim):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = (masked_sums <1e-4)        
        masked_sums += zeros.float()
        return masked_exps/masked_sums

    def forward(self, values, tvmap, rvmaps):

        B, C, T, H, W = values.size()
        # t_feat: target feature
        t_feat = values[:, :, 0]
        # r_feats: refetence features
        r_feats = values[:, :, 1:]
        
        B, Cv, T, H, W = r_feats.size()
        # vmap: visibility map
        # tvmap: target visibility map
        # rvmap: reference visibility map
        # gs: cosine similarity
        # c_m: c_match
        gs_,vmap_ = [], []
        tvmap_t = (F.upsample(tvmap, size=(H, W), mode='bilinear', align_corners=False)>0.5).float()
        for r in range(T):
            rvmap_t = (F.upsample(rvmaps[:,:,r], size=(H, W), mode='bilinear', align_corners=False)>0.5).float()
            # vmap: visibility map
            vmap = tvmap_t * rvmap_t
            gs = (vmap * t_feat * r_feats[:,:,r]).sum(-1).sum(-1).sum(-1)
            #valid sum
            v_sum = vmap[:,0].sum(-1).sum(-1)
            zeros = (v_sum <1e-4)
            gs[zeros] = 0
            v_sum += zeros.float()
            gs = gs / v_sum / C
            gs = torch.ones(t_feat.shape).float().cuda() * gs.view(B,1,1,1)
            gs_.append(gs)
            vmap_.append(rvmap_t)

        gss = torch.stack(gs_, dim=2)
        vmaps = torch.stack(vmap_, dim=2)
        
        #weighted pixelwise masked softmax
        c_match = self.masked_softmax(gss, vmaps, dim=2)
        c_out = torch.sum(r_feats * c_match, dim=2)

        # c_mask
        c_mask = (c_match * vmaps)
        c_mask = torch.sum(c_mask,2)
        c_mask = 1. - (torch.mean(c_mask, 1, keepdim=True))

        return torch.cat([t_feat, c_out, c_mask], dim=1), c_mask


class CPNet(nn.Module):
    def __init__(self, mode='Train'):
        super(CPNet, self).__init__()
        self.A_Encoder = A_Encoder()  # Align 
        self.A_Regressor = A_Regressor() # output: alignment network

        self.Encoder = Encoder()  # Merge
        self.CM_Module = CM_Module()

        self.Decoder = Decoder() 
        
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('mean3d', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1))

    
    def encoding(self, frames, holes):

        batch_size, _, num_frames, height, width = frames.size()
        # padding
        (frames, holes), pad = pad_divide_by([frames, holes], 8, (frames.size()[3], frames.size()[4]))
        
        feat_ = []
        for t in range(num_frames):
            feat = self.A_Encoder(frames[:,:,t], holes[:,:,t])
            feat_.append(feat)
        feats = torch.stack(feat_, dim=2)
        return feats

    def inpainting(self, rfeats, rframes, rholes, frame, hole, gt): 

        batch_size, _, height, width = frame.size() # B C H W
        num_r = rfeats.size()[2] # # of reference frames

        # padding
        (rframes, rholes, frame, hole, gt), pad = pad_divide_by([rframes, rholes, frame, hole, gt], 8, (height, width))
        
        # Target embedding
        tfeat = self.A_Encoder(frame, hole)
        
        # c_feat: Encoder(Copy Network) features
        c_feat_ = [self.Encoder(frame, hole)]
        L_align = torch.zeros_like(frame)
        
        # aligned_r: aligned reference frames
        aligned_r_ = []

        # rvmap: aligned reference frames valid maps
        rvmap_ = []
        
        for r in range(num_r):
            theta_rt = self.A_Regressor(tfeat, rfeats[:,:,r])
            grid_rt = F.affine_grid(theta_rt, frame.size())

            # aligned_r: aligned reference frame
            # reference frame affine transformation
            aligned_r = F.grid_sample(rframes[:,:,r], grid_rt)
            
            # aligned_v: aligned reference visiblity map
            # reference mask affine transformation
            aligned_v = F.grid_sample(1-rholes[:,:,r], grid_rt)
            aligned_v = (aligned_v>0.5).float()

            aligned_r_.append(aligned_r)

            #intersection of target and reference valid map
            trvmap = (1-hole) * aligned_v
            # compare the aligned frame - target frame 
            
            c_feat_.append(self.Encoder(aligned_r, aligned_v))
            
            rvmap_.append(aligned_v)

        aligned_rs = torch.stack(aligned_r_, 2)
        
        c_feats =torch.stack(c_feat_, dim=2)
        rvmaps = torch.stack(rvmap_, dim=2)

        # p_in: paste network input(target features + c_out + c_mask)
        p_in, c_mask = self.CM_Module(c_feats, 1-hole, rvmaps)
        
        pred = self.Decoder(p_in)
        
        _, _, _, H, W = aligned_rs.shape
        c_mask = (F.upsample(c_mask, size=(H, W), mode='bilinear', align_corners=False)).detach()

        comp = pred * (hole) + gt * (1.-hole)


        if pad[2]+pad[3] > 0:
            comp = comp[:,:,pad[2]:-pad[3],:]

        if pad[0]+pad[1] > 0:
            comp = comp[:,:,:,pad[0]:-pad[1]]
            
        comp = torch.clamp(comp, 0, 1)

        return comp


    def forward(self, *args, **kwargs):
        if len(args) == 2:
            return self.encoding(*args)
        else:
            return self.inpainting(*args, **kwargs)