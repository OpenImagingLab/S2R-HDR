#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
sys.path.append('.')
import random
import torch
import numpy as np
from utils.utils import *
from dataset.dataset_sig17 import Img_Dataset, TTA_Img_Dataset
from torch.utils.data import Dataset


class ImgNoGTDataset(Img_Dataset):
    def __init__(self, scene, ldr_path, label_path, exposure_path, patch_size):
        self.ldr_images = read_images(ldr_path)
        # print(ldr_path, label_path)
        self.label = np.zeros_like(self.ldr_images[0])
        self.ldr_patches = self.get_ordered_patches(patch_size)
        self.expo_times = read_expo_times(exposure_path)
        self.patch_size = patch_size
        self.result = []
        self.scene = scene

def TestNoGTDataset(root_dir, patch_size):
    scenes_dir = osp.join(root_dir)
    scenes_list = sorted(os.listdir(scenes_dir))
    ldr_list = []
    label_list = []
    expo_times_list = []
    scene_name_list = []
    for scene in range(len(scenes_list)):
        if not os.path.isdir(os.path.join(scenes_dir, scenes_list[scene])):
            continue
        exposure_file_path = os.path.join(scenes_dir, scenes_list[scene], 'exposure.txt')
        ldr_file_path = list_all_files_sorted(os.path.join(scenes_dir, scenes_list[scene]), '.tif') + \
            list_all_files_sorted(os.path.join(scenes_dir, scenes_list[scene]), '.tiff')
        label_path = os.path.join(scenes_dir, scenes_list[scene])
        expo_times_list += [exposure_file_path]
        ldr_list += [ldr_file_path]
        label_list += [label_path]
        scene_name_list += [scenes_list[scene]]
    for scene, ldr_dir, label_dir, expo_times_dir in zip(scene_name_list, ldr_list, label_list, expo_times_list):
        yield ImgNoGTDataset(scene, ldr_dir, label_dir, expo_times_dir, patch_size)


class SAFNet_Test_NoGT_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.is_training = False

        self.scenes_dir = root_dir
        self.scenes_list = sorted(os.listdir(self.scenes_dir))
        print(self.scenes_list)
        self.image_list = []
        for scene in range(len(self.scenes_list)):
            if not os.path.isdir(os.path.join(self.scenes_dir, self.scenes_list[scene])):
                continue
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path, f"{self.scenes_list[scene]}"]]
        
    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        # label = read_label(self.image_list[index][2], 'HDRImg.hdr') # 'label.hdr' for cropped training data
        scene_name = self.image_list[index][3]
        ldr_images0, ldr_images1, ldr_images2 = ldr_images[0], ldr_images[1], ldr_images[2]

        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images0, expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images1, expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images2, expoTimes[2], 2.2)

        pre_img0 = np.concatenate((pre_img0, ldr_images0), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images1), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images2), 2)

        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        # label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        # label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            # 'label': None,
            'exp': torch.tensor(expoTimes[:3], dtype=torch.float),
            "scene": scene_name,
            'ldr_path':self.image_list[index][1],
            'label_path':self.image_list[index][2],
            'exposure_path':self.image_list[index][0],
            }
        return sample

    def __len__(self):
        return len(self.image_list)



class TTA_Img_Dataset_NoGT(TTA_Img_Dataset):
    def __init__(self, scene, ldr_path, exposure_path, patch_size, num_patches, label_path=None, random_crop=False):
        self.ldr_images = read_images(ldr_path)
        self.label = np.zeros_like(self.ldr_images[0])
        # self.ldr_patches = self.get_ordered_patches(patch_size)
        self.expo_times = read_expo_times(exposure_path)
        self.patch_size = patch_size
        self.result = []
        self.scene = scene
        self.num_patches = num_patches
        self.random_crop = random_crop
        if not self.random_crop:  # using ordered patch as Img_Dataset
            self.ldr_patches = self.get_ordered_patches(patch_size)
            self.num_patches = len(self.ldr_patches)


if __name__ == "__main__":
    data = TestNoGTDataset(root_dir="xxxx/Real-Blur-HDR", patch_size=256)
    scene, item1 = next(data)[12]
    print(scene)
    print(item1.keys())

    def process_image(x, tonemap=False, gamma=False, gamma_value=2.2, save_path=None):
        x = x.transpose(1, 2, 0)
        if tonemap:
            x = (np.log(1 + 5000 * x)) / np.log(1 + 5000)
        elif gamma:
            x = np.power(x, 1/ gamma_value)
        x = x.clip(0.0, 1.0)
        if save_path is not None:
            x = (x * 255).astype(np.uint8)
            cv2.imwrite(save_path, x[..., ::-1])
        return x
    print(item1['input0'].shape, type(item1['input0']))

    save_dir = "tmp/real"
    os.makedirs(save_dir, exist_ok=True)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 4, 1); plt.imshow(process_image(item1['input0'].numpy()[3:,:,:])); plt.title('ldr_0')
    plt.subplot(1, 4, 2); plt.imshow(process_image(item1['input1'].numpy()[3:,:,:])); plt.title('ldr_1')
    plt.subplot(1, 4, 3); plt.imshow(process_image(item1['input2'].numpy()[3:,:,:])); plt.title('ldr_2')
    # plt.subplot(1, 4, 4); plt.imshow(process_image(item1['label'])); plt.title('hdr_gt_0')
    plt.savefig(f"{save_dir}/real-data-val.png")
    