#-*- coding:utf-8 -*-  
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import glob
import cv2
import math
import imageio
from math import log10
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def ldr_to_hdr(imgs, expo, gamma=2.2):
    return (imgs ** gamma) / (expo + 1e-8)

def hdr_to_ldr(hdr_imgs, expo, gamma=2.2):
    # Reverse the exposure adjustment
    ldr_imgs = hdr_imgs * (expo + 1e-8)
    # Apply gamma correction
    return ldr_imgs ** (1 / gamma)


class ExpAug():
    def __init__(self, param, expo_time):
        self.param = torch.tensor(param).to('cuda')
        self.param = torch.exp(self.param * np.log(2))
        self.expo_time = expo_time
        # -2 2

    def transform(self, img, id_ldr=0):
        hdr_img = img[:, :3, :, :]  # First three channels (HDR)
        ldr_img = img[:, 3:, :, :]  # Last three channels (LDR)
        hdr_img = hdr_img * self.param
        ldr_img_linear = ldr_to_hdr(ldr_img, self.expo_time[id_ldr])
        ldr_img_linear = ldr_img_linear * self.param
        ldr_img = hdr_to_ldr(ldr_img_linear, self.expo_time[id_ldr])
        return torch.cat((hdr_img, ldr_img), dim=1)

    def transform_back(self, img):
        return img / self.param



class WBAug():
    def __init__(self, param, expo_time): # para is [1,1.1,0.9]
        self.param = torch.tensor(param).to('cuda')
        # self.range_r = math.exp(self.log_wb_range)
        # self.range_l = 1 - (self.range_l - 1)
        self.init_params = [1., 1., 1.]
        self.expo_time = expo_time

    def transform(self, img, id_ldr=0):
        hdr_img = img[:, :3, :, :]  # First three channels (HDR)
        ldr_img = img[:, 3:, :, :]  # Last three channels (LDR)
        hdr_img = hdr_img * self.param.view(1, 3, 1, 1)
        ldr_img_linear = ldr_to_hdr(ldr_img, self.expo_time[id_ldr])
        ldr_img_linear = ldr_img_linear * self.param.view(1, 3, 1, 1)
        ldr_img = hdr_to_ldr(ldr_img_linear, self.expo_time[id_ldr])
        return torch.cat((hdr_img, ldr_img), dim=1)

    def transform_back(self, img):
        return img / (self.param.view(1, 3, 1, 1))
    

class PermAug():
    def __init__(self):
        # Create a random permutation for the channels
        self.perm = self._generate_permutation().to('cuda')

    def _generate_permutation(self):
        # Generate a permutation for the channels ensuring the two images are treated separately
        base_perm = torch.randperm(3)  # Random permutation for the first three channels
        perm = torch.cat([base_perm, base_perm + 3])  # Append the same permutation for the second image
        return perm

    def transform(self, img, id_ldr=0):
        # img shape: (b, 6, h, w)
        permuted_img = img[:, self.perm, :, :]
        return permuted_img

    def transform_back(self, permuted_img):
        # Create an inverse permutation
        inverse_perm = torch.argsort(self.perm[:3])  # transfer back the label output
        original_img = permuted_img[:, inverse_perm, :, :]
        return original_img


class FlipAug():
    def __init__(self, flip_type=0):
        # Randomly choose whether to flip horizontally, vertically, or not at all
        self.flip_type = flip_type  # 0 or 1

    def transform(self, img, id_ldr=0):
        # img shape: (b, 6, h, w)
        if self.flip_type == 0:
            return torch.flip(img, dims=[3])  # Flip horizontally
        elif self.flip_type == 1:
            return torch.flip(img, dims=[2])  # Flip vertically
        else:
            return img  # No flip

    def transform_back(self, flipped_img):
        # Since flip is symmetric, the same operation can be applied to restore
        if self.flip_type == 0:
            return torch.flip(flipped_img, dims=[3])  # Flip back horizontally
        elif self.flip_type == 1:
            return torch.flip(flipped_img, dims=[2])  # Flip back vertically
        else:
            return flipped_img  # No flip to restore

