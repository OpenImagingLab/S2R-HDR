#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('.')
sys.path.append('..')
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *
from dataset.data_utils import random_crop_v4, random_flip_lrud_v4
from tqdm import tqdm
import copy


class SIG17_Training_Dataset(Dataset):

    def __init__(self, root_dir, sub_set, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.sub_set = sub_set

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], 'label.hdr') # 'label.hdr' for cropped training data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label,
            'exp': torch.tensor(expoTimes[:3], dtype=torch.float),
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)

class SIG17_Training_Dataset_Cache(Dataset):

    def __init__(self, root_dir, sub_set, patch_size=128, cache=False):
        self.root_dir = root_dir
        self.is_training = True
        self.sub_set = sub_set
        self.patch_size = patch_size
        self.cache = cache

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            if not os.path.isdir(os.path.join(self.scenes_dir, self.scenes_list[scene])):
                continue
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path, f"{self.scenes_list[scene]}"]]

        if self.cache:
            self.cache_data()
        if self.patch_size == 512:
            self.image_list = self.image_list * 100
        else:
            self.image_list = self.image_list * int(1500*1000/self.patch_size/self.patch_size)

    def cache_data(self):
        self.data_dict = {}
        print("Caching dataset...")
        for image_list in tqdm(self.image_list):
            expoTimes = read_expo_times(image_list[0])
            # Read LDR images
            ldr_images0, ldr_images1, ldr_images2 = read_images(image_list[1])
            # Read HDR label
            label = read_label(image_list[2], "HDRImg.hdr")
            self.data_dict[image_list[3]] = {"exp": expoTimes,
                                             "ldr0": ldr_images0,
                                             "ldr1": ldr_images1,
                                             "ldr2": ldr_images2,
                                             "label": label
                                             }
        print(f"Caching dataset sucessful!")

    def __getitem__(self, index):
        if self.cache:
            k = self.image_list[index][3]
            ldr_images0 = copy.deepcopy(self.data_dict[k]["ldr0"])
            ldr_images1 = copy.deepcopy(self.data_dict[k]["ldr1"])
            ldr_images2 = copy.deepcopy(self.data_dict[k]["ldr2"])
            expoTimes = copy.deepcopy(self.data_dict[k]["exp"])
            label = copy.deepcopy(self.data_dict[k]["label"])
        else:
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])
            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            # Read HDR label
            label = read_label(self.image_list[index][2], "HDRImg.hdr")
            ldr_images0, ldr_images1, ldr_images2 = ldr_images[0], ldr_images[1], ldr_images[2]
        
        if self.is_training:
            label, ldr_images0, ldr_images1, ldr_images2 = random_crop_v4(label, ldr_images0, ldr_images1, ldr_images2,
                                                                          [self.patch_size, self.patch_size])
            label, ldr_images0, ldr_images1, ldr_images2 = random_flip_lrud_v4(label, ldr_images0, ldr_images1, ldr_images2)
            color_permute = np.random.permutation(3)
            label = label[:,:, color_permute]
            ldr_images0 = ldr_images0[:,:, color_permute]
            ldr_images1 = ldr_images1[:,:, color_permute]
            ldr_images2 = ldr_images2[:,:, color_permute]
        
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
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label,
            'exp': torch.tensor(expoTimes[:3], dtype=torch.float),
            }
        return sample

    def __len__(self):
        return len(self.image_list)


class SIG17_Validation_Dataset(Dataset):

    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # sample dir
        self.scenes_dir = osp.join(root_dir, 'Test')
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], 'HDRImg.hdr') # 'HDRImg.hdr' for test data
        # ldr images process
        pre_img0 = ldr_to_hdr(ldr_images[0], expoTimes[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expoTimes[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expoTimes[2], 2.2)

        # concat: linear domain + ldr domain
        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        if self.crop:
            x = 0
            y = 0
            img0 = pre_img0[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
            label = label[x:x + self.crop_size, y:y + self.crop_size].astype(np.float32).transpose(2, 0, 1)
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)


class Img_Dataset(Dataset):
    def __init__(self, scene, ldr_path, label_path, exposure_path, patch_size):
        self.ldr_images = read_images(ldr_path)
        self.label = read_label(label_path, 'HDRImg.hdr') #  challenge123 hdr_img.hdr
        self.ldr_patches = self.get_ordered_patches(patch_size)
        self.expo_times = read_expo_times(exposure_path)
        self.patch_size = patch_size
        self.result = []
        self.scene = scene

        self.ldr_path = ldr_path
        self.label_path = label_path
        self.exposure_path = exposure_path

    def __getitem__(self, index):
        pre_img0 = ldr_to_hdr(self.ldr_patches[index][0], self.expo_times[0], 2.2)
        pre_img1 = ldr_to_hdr(self.ldr_patches[index][1], self.expo_times[1], 2.2)
        pre_img2 = ldr_to_hdr(self.ldr_patches[index][2], self.expo_times[2], 2.2)
        pre_img0 = np.concatenate((pre_img0, self.ldr_patches[index][0]), 2)
        pre_img1 = np.concatenate((pre_img1, self.ldr_patches[index][1]), 2)
        pre_img2 = np.concatenate((pre_img2, self.ldr_patches[index][2]), 2)
        img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
        img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
        img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2}
        return self.scene, sample

    def get_ordered_patches(self, patch_size):
        ldr_patch_list = []
        h, w, c = self.label.shape
        n_h = h // patch_size + 1
        n_w = w // patch_size + 1
        tmp_h = n_h * patch_size
        tmp_w = n_w * patch_size
        tmp_label = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr0 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr1 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr2 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_label[:h, :w] = self.label
        tmp_ldr0[:h, :w] = self.ldr_images[0]
        tmp_ldr1[:h, :w] = self.ldr_images[1]
        tmp_ldr2[:h, :w] = self.ldr_images[2]

        for x in range(n_w):
            for y in range(n_h):
                if (x+1) * patch_size <= tmp_w and (y+1) * patch_size <= tmp_h:
                    temp_patch_ldr0 = tmp_ldr0[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    temp_patch_ldr1 = tmp_ldr1[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    temp_patch_ldr2 = tmp_ldr2[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    ldr_patch_list.append([temp_patch_ldr0, temp_patch_ldr1, temp_patch_ldr2])

        assert len(ldr_patch_list) == n_h * n_w
        return ldr_patch_list

    def __len__(self):
        return len(self.ldr_patches)

    def rebuild_result(self):
        h, w, c = self.label.shape
        n_h = h // self.patch_size + 1
        n_w = w // self.patch_size + 1
        tmp_h = n_h * self.patch_size
        tmp_w = n_w * self.patch_size
        pred = np.empty((c, tmp_h, tmp_w), dtype=np.float32)

        for x in range(n_w):
            for y in range(n_h):
                pred[:, y*self.patch_size:(y+1)*self.patch_size, x*self.patch_size:(x+1)*self.patch_size] = self.result[x*n_h+y]
        return pred[:, :h, :w], self.label.transpose(2, 0, 1)

    def update_result(self, tensor):
        self.result.append(tensor)



class TTA_Img_Dataset(Dataset):
    def __init__(self, scene, ldr_path, label_path, exposure_path, patch_size, num_patches, random_crop=False):
        self.ldr_images = read_images(ldr_path)
        self.label = read_label(label_path, 'HDRImg.hdr') #  challenge123 hdr_img.hdr
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

    def __getitem__(self, index):
        expoTimes = self.expo_times
        if self.random_crop:
            label = self.label
            ldr_images0 = copy.deepcopy(self.ldr_images[0])
            ldr_images1 = copy.deepcopy(self.ldr_images[1])
            ldr_images2 = copy.deepcopy(self.ldr_images[2])

            label, ldr_images0, ldr_images1, ldr_images2 = random_crop_v4(label, ldr_images0, ldr_images1, ldr_images2,
                                                                            [self.patch_size, self.patch_size])

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
            label = label.astype(np.float32).transpose(2, 0, 1)
            img0 = torch.from_numpy(img0)
            img1 = torch.from_numpy(img1)
            img2 = torch.from_numpy(img2)
            label = torch.from_numpy(label)
        else:
            pre_img0 = ldr_to_hdr(self.ldr_patches[index][0], self.expo_times[0], 2.2)
            pre_img1 = ldr_to_hdr(self.ldr_patches[index][1], self.expo_times[1], 2.2)
            pre_img2 = ldr_to_hdr(self.ldr_patches[index][2], self.expo_times[2], 2.2)
            pre_img0 = np.concatenate((pre_img0, self.ldr_patches[index][0]), 2)
            pre_img1 = np.concatenate((pre_img1, self.ldr_patches[index][1]), 2)
            pre_img2 = np.concatenate((pre_img2, self.ldr_patches[index][2]), 2)
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            img0 = torch.from_numpy(img0)
            img1 = torch.from_numpy(img1)
            img2 = torch.from_numpy(img2)
            label = None
        
        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label,
            'exp': torch.tensor(expoTimes[:3], dtype=torch.float),
            }
        return sample

    def __len__(self):
        return self.num_patches
    
    def get_ordered_patches(self, patch_size):
        ldr_patch_list = []
        h, w, c = self.label.shape
        n_h = h // patch_size + 1
        n_w = w // patch_size + 1
        tmp_h = n_h * patch_size
        tmp_w = n_w * patch_size
        tmp_label = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr0 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr1 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_ldr2 = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
        tmp_label[:h, :w] = self.label
        tmp_ldr0[:h, :w] = copy.deepcopy(self.ldr_images[0])
        tmp_ldr1[:h, :w] = copy.deepcopy(self.ldr_images[1])
        tmp_ldr2[:h, :w] = copy.deepcopy(self.ldr_images[2])

        for x in range(n_w):
            for y in range(n_h):
                if (x+1) * patch_size <= tmp_w and (y+1) * patch_size <= tmp_h:
                    temp_patch_ldr0 = tmp_ldr0[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    temp_patch_ldr1 = tmp_ldr1[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    temp_patch_ldr2 = tmp_ldr2[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
                    ldr_patch_list.append([temp_patch_ldr0, temp_patch_ldr1, temp_patch_ldr2])

        assert len(ldr_patch_list) == n_h * n_w
        return ldr_patch_list

    def rebuild_result(self):
        h, w, c = self.label.shape
        n_h = h // self.patch_size + 1
        n_w = w // self.patch_size + 1
        tmp_h = n_h * self.patch_size
        tmp_w = n_w * self.patch_size
        pred = np.empty((c, tmp_h, tmp_w), dtype=np.float32)

        for x in range(n_w):
            for y in range(n_h):
                pred[:, y*self.patch_size:(y+1)*self.patch_size, x*self.patch_size:(x+1)*self.patch_size] = self.result[x*n_h+y]
        return pred[:, :h, :w], self.label.transpose(2, 0, 1)

    def update_result(self, tensor):
        self.result.append(tensor)



def SIG17_Test_Dataset(root_dir, patch_size):
    scenes_dir = osp.join(root_dir, 'Test')
    scenes_list = sorted(os.listdir(scenes_dir))
    ldr_list = []
    label_list = []
    expo_times_list = []
    scene_name_list = []
    for scene in range(len(scenes_list)):
        exposure_file_path = os.path.join(scenes_dir, scenes_list[scene], 'exposure.txt')
        ldr_file_path = list_all_files_sorted(os.path.join(scenes_dir, scenes_list[scene]), '.tif')
        label_path = os.path.join(scenes_dir, scenes_list[scene])
        expo_times_list += [exposure_file_path]
        ldr_list += [ldr_file_path]
        label_list += [label_path]
        scene_name_list += [scenes_list[scene]]
    for scene, ldr_dir, label_dir, expo_times_dir in zip(scene_name_list, ldr_list, label_list, expo_times_list):
        yield Img_Dataset(scene, ldr_dir, label_dir, expo_times_dir, patch_size)


def Challenge123_Test_Dataset(root_dir, patch_size):
    scenes_dir = osp.join(root_dir, 'Test')
    scenes_list = sorted(os.listdir(scenes_dir))
    scenes_list = sorted(list(set(s[:3] for s in scenes_list)))
    ldr_list = []
    label_list = []
    expo_times_list = []
    scene_name_list = []
    for scene in range(len(scenes_list)):
        exposure_file_path = os.path.join(scenes_dir, scenes_list[scene] + '_2/exposure.txt')
        ldr_file_path = [os.path.join(scenes_dir, scenes_list[scene]+'_1/ldr_img_1.tif'), 
                         os.path.join(scenes_dir, scenes_list[scene]+'_2/ldr_img_2.tif'), 
                         os.path.join(scenes_dir, scenes_list[scene]+'_3/ldr_img_3.tif')]
        label_path = os.path.join(scenes_dir, scenes_list[scene] + '_2') # /HDRImg.hdr
        expo_times_list += [exposure_file_path]
        ldr_list += [ldr_file_path]
        label_list += [label_path]
        scene_name_list += [scenes_list[scene]]
    for scene, ldr_dir, label_dir, expo_times_dir in zip(scene_name_list, ldr_list, label_list, expo_times_list):
        yield Img_Dataset(scene, ldr_dir, label_dir, expo_times_dir, patch_size)


class Challenge123_Training_Dataset_Cache(Dataset):

    def __init__(self, root_dir, sub_set, patch_size=128, cache=False):
        self.root_dir = root_dir
        self.is_training = True
        self.sub_set = sub_set
        self.patch_size = patch_size
        self.cache = cache

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))
        self.scenes_list = sorted(list(set(s[:3] for s in self.scenes_list)))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            scenes_dir = self.scenes_dir
            scene_name = self.scenes_list[scene]
            for short, mid, long in ((1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)):
                ldr_file_path = [os.path.join(scenes_dir, scene_name+f'_{short}/ldr_img_1.tif'),
                                os.path.join(scenes_dir, scene_name+f'_{mid}/ldr_img_2.tif'),
                                os.path.join(scenes_dir, scene_name+f'_{long}/ldr_img_3.tif')]
                exposure_file_path = os.path.join(scenes_dir, scene_name+f'_{mid}/exposure.txt')
                label_path = os.path.join(scenes_dir, scene_name+f'_{mid}')
                self.image_list += [[exposure_file_path, ldr_file_path, label_path, f"{scene_name}_{short}{mid}{long}"]]
        if self.cache:
            self.cache_data()
        if self.patch_size == 512:
            self.image_list = self.image_list * 100
        else:
            self.image_list = self.image_list * int(1500*1000/self.patch_size/self.patch_size / 2)

    def cache_data(self):
        self.data_dict = {}
        print("Caching dataset...")
        for image_list in tqdm(self.image_list):
            expoTimes = read_expo_times(image_list[0])
            # Read LDR images
            ldr_images0, ldr_images1, ldr_images2 = read_images(image_list[1])
            # Read HDR label
            label = read_label(image_list[2], "HDRImg.hdr")
            self.data_dict[image_list[3]] = {"exp": expoTimes,
                                             "ldr0": ldr_images0,
                                             "ldr1": ldr_images1,
                                             "ldr2": ldr_images2,
                                             "label": label
                                             }
        print(f"Caching dataset sucessful!")

    def __getitem__(self, index):
        if self.cache:
            k = self.image_list[index][3]
            ldr_images0 = copy.deepcopy(self.data_dict[k]["ldr0"])
            ldr_images1 = copy.deepcopy(self.data_dict[k]["ldr1"])
            ldr_images2 = copy.deepcopy(self.data_dict[k]["ldr2"])
            expoTimes = copy.deepcopy(self.data_dict[k]["exp"])
            label = copy.deepcopy(self.data_dict[k]["label"])
        else:
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])
            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            # Read HDR label
            label = read_label(self.image_list[index][2], "HDRImg.hdr")
            ldr_images0, ldr_images1, ldr_images2 = ldr_images[0], ldr_images[1], ldr_images[2]
        
        if self.is_training:
            label, ldr_images0, ldr_images1, ldr_images2 = random_crop_v4(label, ldr_images0, ldr_images1, ldr_images2,
                                                                          [self.patch_size, self.patch_size])
            label, ldr_images0, ldr_images1, ldr_images2 = random_flip_lrud_v4(label, ldr_images0, ldr_images1, ldr_images2)
            color_permute = np.random.permutation(3)
            label = label[:,:, color_permute]
            ldr_images0 = ldr_images0[:,:, color_permute]
            ldr_images1 = ldr_images1[:,:, color_permute]
            ldr_images2 = ldr_images2[:,:, color_permute]

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
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)

        sample = {
            'input0': img0, 
            'input1': img1, 
            'input2': img2, 
            'label': label,
            'exp': torch.tensor(expoTimes[:3], dtype=torch.float),
            }
        return sample

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    data = SIG17_Training_Dataset_Cache(root_dir="xxxx/datasets/ImageHDR/sig17", sub_set="Training",
                                        patch_size=512, cache=False)
    
    item1 = data[0]
    print(item1.keys()) # 'hdrs', 'ldrs', 'expos', 'flow_gts', 'flow_mask'

    def process_image(x, tonemap=False, gamma=False, gamma_value=2.2, save_path=None):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
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
    print(item1['input0'].shape)

    save_dir = "tmp/challenge"
    os.makedirs(save_dir, exist_ok=True)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 4, 1); plt.imshow(process_image(item1['input0'][3:,:,:], save_path=f"{save_dir}/ldr_0.png")); plt.title('ldr_0')
    plt.subplot(1, 4, 2); plt.imshow(process_image(item1['input1'][3:,:,:], save_path=f"{save_dir}/ldr_1.png")); plt.title('ldr_1')
    plt.subplot(1, 4, 3); plt.imshow(process_image(item1['input2'][3:,:,:], save_path=f"{save_dir}/ldr_2.png")); plt.title('ldr_2')
    plt.subplot(1, 4, 4); plt.imshow(process_image(item1['label'], tonemap=True, save_path=f"{save_dir}/hdr_gt_0.png")); plt.title('hdr_gt_0')
    plt.savefig(f"{save_dir}/challenge-data-val.png")
    