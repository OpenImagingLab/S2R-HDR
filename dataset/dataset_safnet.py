#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
sys.path.append('.')
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *
from dataset.data_utils import random_crop_v4, random_flip_lrud_v4


class SAFNet_SIG17_Training_Dataset(Dataset):

    def __init__(self, root_dir, sub_set, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.sub_set = sub_set
        self.patch_size = 512

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            if not os.path.isdir(os.path.join(self.scenes_dir, self.scenes_list[scene])):
                continue
            exposure_file_path = os.path.join(self.scenes_dir, self.scenes_list[scene], 'exposure.txt')
            ldr_file_path = list_all_files_sorted(os.path.join(self.scenes_dir, self.scenes_list[scene]), '.tif')
            label_path = os.path.join(self.scenes_dir, self.scenes_list[scene])
            self.image_list += [[exposure_file_path, ldr_file_path, label_path, self.scenes_list[scene]]]
        if is_training:
            self.image_list = self.image_list * 100

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], 'HDRImg.hdr') # 'label.hdr' for cropped training data
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
            'scene':self.image_list[index][3],
            'ldr_path':self.image_list[index][1],
            'label_path':self.image_list[index][2],
            'exposure_path':self.image_list[index][0],
            }
        return sample

    def __len__(self):
        return len(self.image_list)


def read_ldr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1])
    img = (img / 2 ** 16).clip(0, 1).astype(np.float32)
    return img

def read_hdr(img_path):
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1]).astype(np.float32)
    return img

def read_expos(txt_path):
    expos = np.power(2, np.loadtxt(txt_path) - min(np.loadtxt(txt_path))).astype(np.float32)
    return expos


class SAFNet_SIGGRAPH17_Test_Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        sequences = sorted(os.listdir(dataset_dir))
        print(len(sequences))
        self.img0_list = []
        self.img1_list = []
        self.img2_list = []
        self.gt_list = []
        self.expos_list = []
        for seq in sequences:
            ldr_list = sorted(glob(os.path.join(dataset_dir, seq, '*.tif')))
            assert len(ldr_list) == 3
            self.img0_list.append(ldr_list[0])
            self.img1_list.append(ldr_list[1])
            self.img2_list.append(ldr_list[2])
            hdr_list = sorted(glob(os.path.join(dataset_dir, seq, '*.hdr')))
            assert len(hdr_list) == 1
            self.gt_list.append(hdr_list[0])
            expo_list = sorted(glob(os.path.join(dataset_dir, seq, '*.txt')))
            assert len(expo_list) == 1
            self.expos_list.append(expo_list[0])

    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, idx):
        img0 = read_ldr(self.img0_list[idx])
        img1 = read_ldr(self.img1_list[idx])
        img2 = read_ldr(self.img2_list[idx])
        gt = read_hdr(self.gt_list[idx])
        expos = read_expos(self.expos_list[idx])

        imgs_ldr = [img0.copy(), img1.copy(), img2.copy()]
        imgs_lin = []
        for i in range(3):
            imgs_lin.append((imgs_ldr[i] ** 2.2) / expos[i])
        return imgs_lin, imgs_ldr, expos, gt.copy()


class SAFNet_Challenge123_Test_Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        sequences = sorted(os.listdir(dataset_dir))
        sequences = sorted(list(set(s[:3] for s in sequences)))
        print(len(sequences))
        self.img0_list = []
        self.img1_list = []
        self.img2_list = []
        self.gt_list = []
        self.expos_list = []
        for seq in sequences:
            self.img0_list.append(os.path.join(dataset_dir, seq + '_1/ldr_img_1.tif'))
            self.img1_list.append(os.path.join(dataset_dir, seq + '_2/ldr_img_2.tif'))
            self.img2_list.append(os.path.join(dataset_dir, seq + '_3/ldr_img_3.tif'))
            self.gt_list.append(os.path.join(dataset_dir, seq + '_2/HDRImg.hdr'))
            self.expos_list.append(os.path.join(dataset_dir, seq + '_2/exposure.txt'))

    def __len__(self):
        return len(self.gt_list)
    
    def __getitem__(self, idx):
        img0 = read_ldr(self.img0_list[idx])
        img1 = read_ldr(self.img1_list[idx])
        img2 = read_ldr(self.img2_list[idx])
        gt = read_hdr(self.gt_list[idx])
        expos = read_expos(self.expos_list[idx])

        imgs_ldr = [img0.copy(), img1.copy(), img2.copy()]
        imgs_lin = []
        for i in range(3):
            imgs_lin.append((imgs_ldr[i] ** 2.2) / expos[i])
        return imgs_lin, imgs_ldr, expos, gt.copy()


class SAFNet_SIG17_Val_Dataset(Dataset):

    def __init__(self, root_dir, sub_set="Test"):
        self.root_dir = root_dir
        self.is_training = False
        self.sub_set = sub_set

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
        
    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], 'HDRImg.hdr') # 'label.hdr' for cropped training data
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
            "scene": scene_name,
            'ldr_path':self.image_list[index][1],
            'label_path':self.image_list[index][2],
            'exposure_path':self.image_list[index][0],
            }
        return sample

    def __len__(self):
        return len(self.image_list)


class SAFNet_Challenge123_Val_Dataset(Dataset):

    def __init__(self, root_dir, sub_set="Test"):
        self.root_dir = root_dir
        self.is_training = False
        self.sub_set = sub_set

        self.scenes_dir = osp.join(root_dir, self.sub_set)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))
        self.scenes_list = sorted(list(set(s[:3] for s in self.scenes_list)))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            scenes_dir = self.scenes_dir
            scene_name = self.scenes_list[scene]
            for short, mid, long in ((1,2,3), ):
                ldr_file_path = [os.path.join(scenes_dir, scene_name+f'_{short}/ldr_img_1.tif'),
                                os.path.join(scenes_dir, scene_name+f'_{mid}/ldr_img_2.tif'),
                                os.path.join(scenes_dir, scene_name+f'_{long}/ldr_img_3.tif')]
                exposure_file_path = os.path.join(scenes_dir, scene_name+f'_{mid}/exposure.txt')
                label_path = os.path.join(scenes_dir, scene_name+f'_{mid}')
                self.image_list += [[exposure_file_path, ldr_file_path, label_path, f"{scene_name}"]]

    def __getitem__(self, index):
        # Read exposure times
        expoTimes = read_expo_times(self.image_list[index][0])
        # Read LDR images
        ldr_images = read_images(self.image_list[index][1])
        # Read HDR label
        label = read_label(self.image_list[index][2], "HDRImg.hdr")
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
            "scene": scene_name,
            'ldr_path':self.image_list[index][1],
            'label_path':self.image_list[index][2],
            'exposure_path':self.image_list[index][0],
            }
        return sample

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    data = SAFNet_SIG17_Training_Dataset(root_dir="xxx/sig17",
                                         sub_set="Training",
                                         is_training=True)
    item1 = data[0]
    print(item1.keys()) # 'hdrs', 'ldrs', 'expos', 'flow_gts', 'flow_mask'

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
    print(item1['input0'].shape)

    save_dir = "tmp/safnet-sig17"
    os.makedirs(save_dir, exist_ok=True)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 4, 1); plt.imshow(process_image(item1['input0'].numpy()[3:,:,:], save_path=f"{save_dir}/ldr_0.png")); plt.title('ldr_0')
    plt.subplot(1, 4, 2); plt.imshow(process_image(item1['input1'].numpy()[3:,:,:], save_path=f"{save_dir}/ldr_1.png")); plt.title('ldr_1')
    plt.subplot(1, 4, 3); plt.imshow(process_image(item1['input2'].numpy()[3:,:,:], save_path=f"{save_dir}/ldr_2.png")); plt.title('ldr_2')
    plt.subplot(1, 4, 4); plt.imshow(process_image(item1['label'].numpy(), tonemap=True, save_path=f"{save_dir}/hdr_gt_0.png")); plt.title('hdr_gt_0')
    plt.savefig(f"{save_dir}/safnet-sig17-data-val.png")
    