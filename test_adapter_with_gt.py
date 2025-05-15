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

from dataset.dataset_sig17 import SIG17_Test_Dataset, Challenge123_Test_Dataset
from dataset.dataset_safnet import SAFNet_Challenge123_Val_Dataset, SAFNet_SIG17_Val_Dataset
from models.hdr_transformer import HDRTransformer
from models.SCTNet import SCTNet, inject_trainable_adapter_transformer
from models.SAFNet import SAFNet, inject_trainable_adapter_cnn
from train import test_single_img
from utils.utils import *

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--data_name", type=str, default="sct", choices=["sig17", "sct", "challenge123", "s2r-hdr"]),
parser.add_argument("--dataset_dir", type=str, default='../../datasets/ImageHDR/sig17',
                        help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=24, metavar='N',
                        help='testing batch size (default: 24)') # 16G 
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--pretrained_model', type=str, default='pretrained_models/pretrained_model.pth')
parser.add_argument('--save_results', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, default="./results/hdr_transformer_official")
parser.add_argument('--model', type=str, default='HDR-transformer', choices=['HDR-transformer', 'SCTNet', "SAFNet"])

# adapter
parser.add_argument('--optim', default="Adam", type=str, help='adapter')
parser.add_argument('--r1', default=1, type=int, help='sim')
parser.add_argument('--r2', default=64, type=int, help='real')
parser.add_argument('--scale1', default=1.0, type=float, help='sim')
parser.add_argument('--scale2', default=1.0, type=float, help='real')
parser.add_argument('--learn_scale', default=False, action='store_true', help='adapter')


def set_scale(update_model, scale1, scale2, device):
    for name, module in update_model.named_modules():
        if hasattr(module, 'scale1'):
            module.scale1 = scale1
        if hasattr(module, 'scale2'):
            module.scale2 = scale2

def main():
    # Settings
    args = parser.parse_args()
    args.save_dir = args.save_dir + f"_b{args.test_batch_size}"

    # pretrained_model
    print(f">>>>>>>>> Start Testing {args.model}>>>>>>>>>")
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
    set_scale(model, scale1=args.scale1, scale2=args.scale2, device=device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()
    if args.model == "SAFNet":
        if args.data_name == "challenge123":
            datasets = SAFNet_Challenge123_Val_Dataset(args.dataset_dir)
        else:
            datasets = SAFNet_SIG17_Val_Dataset(args.dataset_dir)
        datasets = DataLoader(dataset=datasets, batch_size=1, num_workers=1, shuffle=False)
    else:
        if args.data_name == "challenge123":
            datasets = Challenge123_Test_Dataset(args.dataset_dir, args.patch_size)
        else:
            datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'test.txt'), 'w+') as f:  
        for idx, img_dataset in enumerate(datasets):
            with torch.no_grad():
                if args.model == "SAFNet":
                    batch_ldr0, batch_ldr1, batch_ldr2 = img_dataset['input0'].to(device), img_dataset['input1'].to(device), \
                        img_dataset['input2'].to(device) # bgr
                    label = img_dataset['label'].to(device) # bgr
                    scene = img_dataset['scene'][0]
                    _, pred_img = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    pred_img = pred_img.detach().cpu().numpy()[0]
                    label = label.detach().cpu().numpy()[0]
                else:
                    pred_img, label, scene = test_single_img(model, img_dataset, device, args.test_batch_size)
            pred_hdr = pred_img.copy() # bgr
            pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1] # rgb
            # psnr-l and psnr-\mu
            scene_psnr_l = compare_psnr(label, pred_img, data_range=1.0)
            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred_img)
            scene_psnr_mu = compare_psnr(label_mu, pred_img_mu, data_range=1.0)
            # ssim-l
            pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
            label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
            scene_ssim_l = calculate_ssim(pred_img, label)
            # ssim-\mu
            pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

            psnr_l.update(scene_psnr_l)
            ssim_l.update(scene_ssim_l)
            psnr_mu.update(scene_psnr_mu)
            ssim_mu.update(scene_ssim_mu)
            f.write(f"{idx}, {scene}, PSNR_mu:{scene_psnr_mu:.2f}, PNSR_l:{scene_psnr_l:.2f}, " \
                    f"SSIM_mu: {scene_ssim_mu:.4f}, SSIM_l: {scene_ssim_l:.4f}\n")

            # save results
            if args.save_results:
                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                cv2.imwrite(os.path.join(args.save_dir, f'{idx}_{scene}_pred.hdr'), pred_hdr[..., ::-1])
                cv2.imwrite(os.path.join(args.save_dir, f'{idx}_{scene}_pred.png'), pred_img_mu.astype(np.uint8))
        print("Average PSNR_mu: {:.2f}  PSNR_l: {:.2f}".format(psnr_mu.avg, psnr_l.avg))
        print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
        print(">>>>>>>>> Finish Testing >>>>>>>>>")
        f.write(f"Average, PSNR_mu:{psnr_mu.avg:.4f}, PNSR_l:{psnr_l.avg:.4f}, " \
                f"SSIM_mu: {ssim_mu.avg:.6f}, SSIM_l: {ssim_l.avg:.6f}\n")
        f.write(f"Average, PSNR_mu:{psnr_mu.avg:.2f}, PNSR_l:{psnr_l.avg:.2f}, " \
                f"SSIM_mu: {ssim_mu.avg:.4f}, SSIM_l: {ssim_l.avg:.4f}\n")


if __name__ == '__main__':
    main()
