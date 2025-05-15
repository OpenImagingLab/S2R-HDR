"""
Dataloader for processing s2r-hdr into training data
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
from imageio import imread
from tqdm import tqdm
import cv2
import random
import pyexr

import sys
sys.path.append('..')
sys.path.append('./')
from torch.utils.data import Dataset
import torch
from dataset.data_utils import random_crop, random_crop_v2, random_flip_lrud, random_flip_lrud_v2, center_crop

np.random.seed(0)


def hdr_to_ldr(img, expo, gamma=2.2, std_min=1e-4, std_max=1e-3, add_noise=False):
    # add noise to low expo
    if add_noise:
        stdv = np.random.rand(*img.shape) * (std_max - std_min) + std_min
        noise = np.random.normal(0, stdv)
        img = (img + noise).clip(0, 1)
    img = np.power(img * expo, 1.0 / gamma)
    img = img.clip(0, 1)
    return img

def read_list(list_path, ignore_head=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    return lists


def process_image(x, tonemap=False, gamma=False, gamma_value=2.2, save_path=None):
    x = x.permute(1, 2, 0).numpy()
    if tonemap:
        x = (np.log(1 + 5000 * x)) / np.log(1 + 5000)
    elif gamma:
        x = np.power(x, 1/ gamma_value)
    x = x.clip(0.0, 1.0)
    if save_path is not None:
        x = (x * 255).astype(np.uint8)
        cv2.imwrite(save_path, x[..., ::-1])
    return x


def discretize_to_xbit(img, bit=16):
    # 12/ 16 bit
    max_int = 2**bit-1
    img = np.clip(img, 0.0, 1.0)
    img_uint16 = np.uint16(img * max_int).astype(np.float64) / max_int
    return img_uint16


def read_hdr_cv2(img_path, scale=1.0):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = img.clip(0.0, float("inf")) / scale
    return img


def read_hdr_pyexr(img_path, scale=1.0):
    with pyexr.open(img_path) as file:
        img = file.get(precision=pyexr.HALF)
    img = img / scale
    return img


def read_flow(file_name):
    flow_exr = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow_exr = cv2.cvtColor(flow_exr, cv2.COLOR_BGR2RGB)
    flow_exr = np.array(flow_exr, dtype=np.float64)
    return flow_exr

def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)

class RenderFlowDataset(Dataset):

    def __init__(self, root_dir, nframes=3, nexps=3, cache=False, task="train", scale=None, scale_value=1.0, inv_gamma=False, patch_size=128, read_fun="pyexr"):
        """
        scale: [None, max, percentile, number]
        scale_value: [1.0, .max(), 99.5, 2.0]
        """
        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps

        self.read_fun = read_fun
        self.ext = ".exr"
        self.begin_frame = 0
        self.end_frame = 23
        self.extra_frame = None # use random value (1, 4)
        self.frames = 24
        self.frame_scale = 1
        self.hdr_scale = 1.0
        self.scale = scale
        self.scale_value = scale_value
        self.inv_gamma = inv_gamma
        self.only_deblur = False
        self.patch_size = patch_size

        self.task = task
        self.is_training = True if task == "train" else False
        if self.is_training:
            list_name = 'trainlist.txt'
            self.repeat = 1
        else:
            list_name = 'testlist.txt'
            self.repeat = 1

        self.patch_list = read_list(os.path.join(self.root_dir, list_name))
        self.cache = cache
        if self.cache:
            self.cache_data(self.patch_list)
        
        if self.is_training:
            if self.patch_size == 512:
                self.patch_list = self.patch_list * (self.end_frame - self.begin_frame) # 8 
            else:
                self.patch_list = self.patch_list * (self.end_frame - self.begin_frame) * 2 # * 30
        else:
            self.patch_list = self.patch_list
    
    def read_hdr(self, x, scale=1.0):
        if self.read_fun == "opencv":
            return read_hdr_cv2(x, scale)
        elif self.read_fun == "pyexr":
            return read_hdr_pyexr(x, scale)
        else:
            raise ValueError(f"Don not support read func({self.read_fun}), opencv or pyexr")

    def get_scale(self, img):
        scale_ = 1.0
        if self.scale == "percentile":
            scale_ = np.percentile(img, self.scale_value)
        elif self.scale == "max":
            scale_ = img.max()
        elif self.scale == "number":
            scale_ = self.scale_value
        elif self.scale is None or self.scale.lower() == "none" or self.scale == "percentile_norm":
            scale_ = 1.0
        else:
            raise ValueError(f"Don't support Scale Type, {self.scale}, {self.scale_value}")
        return scale_

    def cache_data(self, patch_list):
        self.data_dict = {}
        print("Caching dataset...")
        for img_dir in tqdm(patch_list):
            img_dir = os.path.join(self.root_dir, img_dir)
            for idx in range(self.begin_frame * self.frame_scale, self.frames * self.frame_scale):
                img_path = os.path.join(img_dir, "img", f'{idx:04d}{self.ext}')
                img = self.read_hdr(img_path, scale=self.hdr_scale)
                scale_ = self.get_scale(img)
                img = img / scale_
                # flow = read_flow(os.path.join(img_dir, "flow", "HDR", f'{idx:04d}{self.ext}'))
                flow = None
                self.data_dict[img_path] = {"img": img, "flow": flow}
        print(f"Caching {self.task} dataset sucessful!")

    def __getitem__(self, index):
        img_dir = os.path.join(self.root_dir, self.patch_list[index])
        if self.is_training:
            self.extra_frame = random.randint(1, 4)
            frame = random.randint(self.begin_frame, self.end_frame - self.nframes - self.extra_frame)
            img_idxs = list(range(frame, frame + self.nframes + self.extra_frame))
            img_idxs = random.sample(img_idxs, self.nframes)
            # img_idxs = sorted(img_idxs)
        else:
            frame = (self.frames - self.nframes) // 2
            img_idxs = list(range(frame, frame + self.nframes*2, 2))

        if self.is_training and np.random.random() > 0.5: # inverse time order
            img_idxs = img_idxs[::-1]
        img_paths = [os.path.join(img_dir, "img", f'{idx:04d}{self.ext}') for idx in img_idxs]
        # print(img_paths)
        if self.nexps == 2:
            exposures = self._get_2exposures(index)
        elif self.nexps == 3:
            exposures = self._get_3exposures(index)
        else:
            raise Exception("Unknow exposures")

        hdrs = []
        for img_path in img_paths:
            if self.cache:
                img = self.data_dict[img_path]["img"].copy()
                flow = None # self.data_dict[img_path]["flow"].copy()
            else:
                img = self.read_hdr(img_path, scale=self.hdr_scale)
                scale_ = self.get_scale(img)
                img = img / scale_
                flow = None # read_flow(img_path.replace("img", "flow"))
            hdrs.append(img)

        crop_h, crop_w, _ = hdrs[0].shape
        if self.is_training:
            crop_h, crop_w = self.patch_size, self.patch_size
            hdrs = random_flip_lrud(hdrs)
            hdrs = random_crop(hdrs, [crop_h, crop_w])
            color_permute = np.random.permutation(3)
            for i in range(len(hdrs)):
                hdrs[i] = hdrs[i][:,:,color_permute]
        else:
            if isinstance(self.patch_size, (tuple, list)):
                crop_h, crop_w = self.patch_size[1], self.patch_size[0]
            else:
                crop_h, crop_w = self.patch_size, self.patch_size
            hdrs = center_crop(hdrs, [crop_h, crop_w])

        hdrs, ldrs = self.re_expose_ldrs(hdrs, exposures)
        
        imgs = []
        for i in range(len(ldrs)):
            img = ldr_to_hdr(ldrs[i], exposures[i], 2.2)
            # concat: linear domain + ldr domain
            img = np.concatenate((img, ldrs[i]), 2).astype(np.float32).transpose(2, 0, 1)
            imgs.append(img)
        label = hdrs[self.nframes // 2].astype(np.float32).transpose(2, 0, 1)
        sample = {
            'input0': imgs[0],
            'input1': imgs[1],
            'input2': imgs[2],
            'label': label,
            "exp": exposures
        }
        return sample

    def _get_2exposures(self, index):
        cur_high = False
        exposures = np.ones(self.nframes, dtype=np.float32)
        high_expo = np.random.choice([4., 8.])

        if cur_high:
            for i in range(0, self.nframes, 2):
                exposures[i] = high_expo
        else:
            for i in range(1, self.nframes, 2):
                exposures[i] = high_expo
        return exposures

    def _get_3exposures(self, index):
        expos = random.choice(((1, 4, 16), (1, 4, 16), (1, 8, 64))) # 1 / 3
        exposures = np.array(expos).astype(np.float32)
        return exposures

    def re_expose_ldrs(self, hdrs, exposures):
        if self.is_training:
            mid = len(hdrs) // 2
            new_hdrs = []
            # if np.random.random() > 0.5:
            #     factor = np.random.uniform(1.0, 4.0)
            #     anchor = hdrs[mid].max()
            #     new_anchor = anchor * factor
            # else:
            #     percent = np.random.uniform(98, 100)
            #     anchor = 1.0
            #     new_anchor = np.percentile(hdrs[mid], percent)
            factor = np.random.uniform(0.25, 4.0) # 0.125, 8.0
            anchor = hdrs[mid].max()
            new_anchor = anchor * factor

            # gamma_value = np.random.uniform(1.0, 2.5)
            for idx, hdr in enumerate(hdrs):
                # if self.inv_gamma:
                #     hdr = np.power(hdr, gamma_value)
                if anchor == 0:
                    new_hdr = hdr
                else:
                    new_hdr = (hdr / (anchor + 1e-8) * new_anchor).clip(0, 1)
                new_hdrs.append(new_hdr)
        else:
            new_hdrs = hdrs
        
        ldrs = self.creat_ldrs(new_hdrs, exposures)
        return new_hdrs, ldrs

    def creat_ldrs(self, hdrs, exposures):
        ldrs = []
        for i in range(len(hdrs)):
            if exposures[i] == 1:
                ldr = hdr_to_ldr(hdrs[i], exposures[i], std_min=1e-4, std_max=1e-3, add_noise=True) # std_min=1e-3, std_max=1e-2
            elif i == 1 and (exposures[i] == 4 or exposures[i] == 8):
                ldr = hdr_to_ldr(hdrs[i], exposures[i], std_min=1e-5, std_max=1e-4, add_noise=True) # std_min=1e-4, std_max=1e-3
            else:
                ldr = hdr_to_ldr(hdrs[i], exposures[i], add_noise=False)
            ldr = discretize_to_xbit(ldr, bit=16)  # 16bit quant
            ldrs.append(ldr)
        return ldrs

    def __len__(self):
        return len(self.patch_list)


def generate_test_dataset():
    import tifffile
    np.random.seed(0)
    random.seed(0)
    data = RenderFlowDataset(root_dir="xxx/S2R-HDR-processed",
                             nframes=3, nexps=3, cache=False, task="val", patch_size=(1500, 1000))
    save_path = "xxx/S2R-HDR-processed-Test"
    os.makedirs(save_path, exist_ok=True)
    data_len = len(data)
    for i in tqdm(range(data_len)):
        if i % 10 != 0:
            continue
        item = data[i]
        exp = item['exp'][:3]
        exp = [np.log2(x) for x in exp]
        # print(f"{exp[0]}\n{exp[1]}\n{exp[2]}")
        os.makedirs(os.path.join(save_path, f"{i:03d}"), exist_ok=True)
        tifffile.imwrite(os.path.join(save_path, f"{i:03d}", "input_1.tif"), data=(item['input0'][3:,:,:] * 65535).astype(np.uint16).transpose(1, 2, 0))
        tifffile.imwrite(os.path.join(save_path, f"{i:03d}", "input_2.tif"), data=(item['input1'][3:,:,:] * 65535).astype(np.uint16).transpose(1, 2, 0))
        tifffile.imwrite(os.path.join(save_path, f"{i:03d}", "input_3.tif"), data=(item['input2'][3:,:,:] * 65535).astype(np.uint16).transpose(1, 2, 0))
        cv2.imwrite(os.path.join(save_path, f"{i:03d}", "HDRImg.hdr"), item['label'].transpose(1, 2, 0)[..., ::-1].clip(0, 1.0))
        with open(os.path.join(save_path, f"{i:03d}", "exposure.txt"), "w") as f:
            f.write(f"{exp[0]}\n{exp[1]}\n{exp[2]}")
        # break

if __name__  == "__main__":
    # generate_test_dataset()
    # exit()
    import time
    data = RenderFlowDataset(root_dir="xxx", 
                                 nframes=3, nexps=3, cache=False, task="train", scale="max", scale_value=1.0)
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

    save_dir = "tmp/render"
    os.makedirs(save_dir, exist_ok=True)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(30, 8))
    plt.subplot(1, 4, 1); plt.imshow(process_image(item1['input0'][3:,:,:], save_path=f"{save_dir}/ldr_0.png")); plt.title('ldr_0')
    plt.subplot(1, 4, 2); plt.imshow(process_image(item1['input1'][3:,:,:], save_path=f"{save_dir}/ldr_1.png")); plt.title('ldr_1')
    plt.subplot(1, 4, 3); plt.imshow(process_image(item1['input2'][3:,:,:], save_path=f"{save_dir}/ldr_2.png")); plt.title('ldr_2')
    plt.subplot(1, 4, 4); plt.imshow(process_image(item1['label'], tonemap=True, save_path=f"{save_dir}/hdr_gt_0.png")); plt.title('hdr_gt_0')
    plt.savefig(f"{save_dir}/render-blur-data-val.png")

