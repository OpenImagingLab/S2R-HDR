#-*- coding:utf-8 -*-  
import os
import os.path as osp
import sys
import time
import glob
import logging
import argparse

from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm

from dataset.dataset_nogt import TestNoGTDataset, SAFNet_Test_NoGT_Dataset
from models.hdr_transformer import HDRTransformer
from models.SCTNet import SCTNet, inject_trainable_adapter_transformer
from models.SAFNet import SAFNet, inject_trainable_adapter_cnn
from train import test_single_img
from train_adapter import set_scale
from utils.utils import *

import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, 
                    default="xxx/datasets/xxx",
                    help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=24, metavar='N',
                        help='testing batch size (default: 24)') # 16G 
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./outputs/test-nogt")
parser.add_argument('--model', type=str, default='SAFNet', choices=['HDR-transformer', 'SCTNet', "SAFNet"])
parser.add_argument('--r1', default=1, type=int, help='sim')
parser.add_argument('--r2', default=64, type=int, help='real')
parser.add_argument('--adapter', action='store_true', default=False)
parser.add_argument('--prefix', default=None, type=str, help='save file prefix')

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def main():
    # Settings
    args = parser.parse_args()
    args.save_dir = args.save_dir + f"_b{args.test_batch_size}"

    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    upscale = 4
    window_size = 8
    height = (256 // upscale // window_size + 1) * window_size
    width = (256 // upscale // window_size + 1) * window_size

    # model architecture
    model_dict = {
        'HDR-transformer': HDRTransformer(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6),
        'SCTNet': SCTNet(img_size=(height, width), in_chans=18,
                            window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                            embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect'),
        "SAFNet": SAFNet(),
    }
    print(f"Selected model: {args.model}")
    model = model_dict[args.model]
    if args.adapter:
        if args.model == "SCTNet":
            inject_trainable_adapter_transformer(model, r1=args.r1, r2=args.r2)
            msg = "Insert adapter ot SCTNet"
            print(msg)
        elif args.model == "SAFNet":
            inject_trainable_adapter_cnn(model, r1=args.r1, r2=args.r2)
            msg = "Insert adapter to SAFNet"
            print(msg)
        else:
            raise ValueError("Current just support SCTNet and SAFNet adapter")
    
    ckpt_dict = torch.load(args.pretrained_model)
    args.scale1 = ckpt_dict.get("scale1", 1.0)
    args.scale2 = ckpt_dict.get("scale2", 1.0)
    if "state_dict" in ckpt_dict:
        ckpt_dict = ckpt_dict['state_dict']
    model.load_state_dict({k.replace('module.',''): v for k, v in ckpt_dict.items()})
    if args.adapter:
        set_scale(model, scale1=args.scale1, scale2=args.scale2, device=device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()
    if args.model == "SAFNet":
        datasets = SAFNet_Test_NoGT_Dataset(args.dataset_dir)
        datasets = DataLoader(dataset=datasets, batch_size=1, num_workers=1, shuffle=False)
    else:
        datasets = TestNoGTDataset(args.dataset_dir, args.patch_size)
    os.makedirs(args.save_dir, exist_ok=True)
    # with open(os.path.join(args.save_dir, 'test.txt'), 'w+') as f:  
    for idx, img_dataset in enumerate(datasets):
        with torch.no_grad():
            if args.model == "SAFNet":
                batch_ldr0, batch_ldr1, batch_ldr2 = img_dataset['input0'].to(device), img_dataset['input1'].to(device), \
                    img_dataset['input2'].to(device) # bgr
                scene = img_dataset['scene'][0]
                print(scene)
                ldrs = [batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()]
                # padder = InputPadder(ldrs[0].shape, divis_by=128)
                # ldrs = padder.pad(ldrs)
                # print(ldrs[0].shape)
                _, pred_img = model(ldrs[0], ldrs[1], ldrs[2])
                # pred_img = padder.unpad(pred_img)
                pred_img = pred_img.detach().cpu().numpy()[0]
            else:
                pred_img, _, scene = test_single_img(model, img_dataset, device, args.test_batch_size)
        pred_hdr = pred_img.copy() # bgr 
        pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1] # rgb
        pred_img_mu = range_compressor(pred_img).transpose(1, 2, 0)
        # save results
        if args.save_results:
            if not osp.exists(args.save_dir):
                os.makedirs(args.save_dir)
            prefix = '_ada' if args.adapter else ""
            prefix = prefix + ('_' + f'{args.prefix}') if args.prefix else ''
            cv2.imwrite(os.path.join(args.save_dir, f'{idx}_{scene}_{args.model}{prefix}.hdr'),
                        pred_hdr[..., ::-1])
            cv2.imwrite(os.path.join(args.save_dir, f'{idx}_{scene}_{args.model}{prefix}.png'),
                        (pred_img_mu.clip(0, 1.0)*255).astype(np.uint8))

if __name__ == '__main__':
    main()




