# -*- coding:utf-8 -*-
import os
import time
import argparse
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset.dataset_sig17 import SIG17_Training_Dataset, SIG17_Validation_Dataset, SIG17_Test_Dataset, \
    Challenge123_Training_Dataset_Cache, SIG17_Training_Dataset_Cache, Challenge123_Test_Dataset
from dataset.dataset_render import RenderFlowDataset
from dataset.dataset_safnet import SAFNet_SIG17_Training_Dataset, SAFNet_Challenge123_Val_Dataset
from models.loss import L1MuLoss, JointReconPerceptualLoss, JointReconGammaPerceptualLoss, SAFNetLoss
from models.hdr_transformer import HDRTransformer
from models.SCTNet import SCTNet
from models.SAFNet import SAFNet
from utils.utils import *

from accelerate import Accelerator
from accelerate.utils import set_seed

accelerator = Accelerator()
device = accelerator.device


def get_args():
    parser = argparse.ArgumentParser(description='ImageHDR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='HDR-Transformer', choices=['HDR-Transformer', 'SCTNet', "SAFNet"])
    parser.add_argument("--data_name", type=str, default="sct", choices=["sig17", "sct", "sct-cha-cache", "challenge123-cache", "sig17-cache", "sct-cache", "s2r-hdr"],
                        help='dataset directory'),
    parser.add_argument("--dataset_dir", type=str, default='../../datasets/ImageHDR/sct',
                        help='dataset directory'),
    parser.add_argument("--dataset_dir2", type=str, default=None,
                        help='dataset2 directory'),
    parser.add_argument("--test_dataset_dir", type=str, default='../../datasets/ImageHDR/sct',
                        help='dataset directory'),
    parser.add_argument('--patch_size', type=int, default=256),
    parser.add_argument("--sub_set", type=str, default='sig17_training_crop128_stride64',
                        help='dataset directory')
    parser.add_argument('--logdir', type=str, default='experiments/hdr-transformer',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    # Training
    parser.add_argument('--resume', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--loss_func', type=int, default=1,
                        help='loss functions for training')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--lr_decay_interval', type=int, default=100,
                        help='decay learning rate every N epochs(default: 100)')
    parser.add_argument('--lr_min', type=float, default=0.0001,
                        help='mini learning rate (default: 0.0001)') # for safnet 0.00001
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='training batch size (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=24, metavar='N',
                        help='testing batch size (default: 24)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                        help='how many epoch to test before training')
    parser.add_argument('--ckpt_interval', type=int, default=1, metavar='N', 
                        help='how many epoch to save ckpt before training status')
    parser.add_argument('--cache', action='store_true', default=False, help='debug')
    parser.add_argument('--scale', default="max", choices=[None, "max", "percentile", "number", "none"], help='scale type')
    parser.add_argument('--scale_value', default=1.0, type=float, help='scale value')
    parser.add_argument('--val', action='store_true', default=False, help='val')
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    
    for batch_idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                            batch_data['input2'].to(device)
        label = batch_data['label'].contiguous().to(device) 
        if args.loss_func == 3:
            pred_m, pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
            loss = criterion(pred, label, pred_m)
        elif args.loss_func == 2:
            pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
            exp = batch_data['exp'].to(device) 
            loss = criterion(pred, label, exp)
        else:
            pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
            loss = criterion(pred, label)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        
        avg_loss = accelerator.gather(loss).mean()
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            if accelerator.is_main_process:
                with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
                    msg = '{} Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\tlr: {:.6f}\t' \
                        'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'\
                        'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        batch_idx,
                        len(train_loader),
                        100. * batch_idx / len(train_loader),
                        avg_loss.item(),
                        scheduler.get_last_lr()[0],
                        batch_time=batch_time,
                        data_time=data_time
                    )
                    print(msg);f.write(msg + '\n')
            if args.debug:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dict = {
            'state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer': accelerator.unwrap_model(optimizer).state_dict(),
            'scheduler': accelerator.unwrap_model(scheduler).state_dict(),
            }
        if epoch % args.ckpt_interval == 0 or epoch == args.epochs:
            torch.save(save_dict, os.path.join(args.logdir, "ckpts", f'epoch_{epoch}.pth'))
        torch.save(save_dict, os.path.join(args.logdir, "ckpts", f'last_ckpt.pth'))


# for evaluation with limited GPU memory
def test_single_img(model, img_dataset, device, batch_size=1):
    dataloader = DataLoader(dataset=img_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    with torch.no_grad():
        for scene, batch_data in tqdm(dataloader, total=len(dataloader)):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), \
                                                 batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            output = model(batch_ldr0, batch_ldr1, batch_ldr2).clip(0.0, 1.0)
            for i in range(output.shape[0]):
                img_dataset.update_result(output[i].detach().cpu().numpy().astype(np.float32))
    pred, label = img_dataset.rebuild_result()
    return pred, label, scene[0]


def test(args, model, device, optimizer, scheduler, epoch, cur_psnr, **kwargs):
    model.eval()
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    if args.model == "SAFNet":
        if "challenge" in args.test_dataset_dir.lower():
            dataset_test = SAFNet_Challenge123_Val_Dataset(args.test_dataset_dir, "Test")
        else:
            dataset_test = SAFNet_SIG17_Training_Dataset(args.test_dataset_dir, "Test", is_training=False)
        test_datasets = DataLoader(dataset=dataset_test, batch_size=1, num_workers=1, shuffle=False)
    else:
        if "challenge" in args.test_dataset_dir.lower():
            test_datasets = Challenge123_Test_Dataset(args.test_dataset_dir, args.patch_size) 
        else:
            test_datasets = SIG17_Test_Dataset(args.test_dataset_dir, args.patch_size) 
    for idx, img_dataset in enumerate(test_datasets):
        with torch.no_grad():
            if args.model == "SAFNet":
                batch_ldr0, batch_ldr1, batch_ldr2 = img_dataset['input0'].to(device), img_dataset['input1'].to(device), \
                    img_dataset['input2'].to(device)
                label = img_dataset['label'].contiguous().to(device) 
                _, pred_img = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                pred_img = pred_img.detach().cpu().numpy()[0]
                label = label.detach().cpu().numpy()[0]
            else:
                pred_img, label, _ = test_single_img(model, img_dataset, device, args.test_batch_size)
        scene_psnr_l = compare_psnr(label, pred_img, data_range=1.0)

        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)

        scene_psnr_mu = compare_psnr(label_mu, pred_img_mu, data_range=1.0)
        pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
        label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)

        scene_ssim_l = calculate_ssim(pred_img, label) # H W C data_range=0-255
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)
        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)
        if args.debug:
            break
    if accelerator.is_main_process:
        with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:            
            msg = '==Testing==\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}'.format(
                psnr_l.avg,
                psnr_mu.avg,
                ssim_l.avg,
                ssim_mu.avg
            )
            print(msg); f.write(msg+"\n")
            if psnr_mu.avg > cur_psnr[0]:
                # save_model
                save_dict = {
                    'state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': accelerator.unwrap_model(optimizer).state_dict(),
                    'scheduler': accelerator.unwrap_model(scheduler).state_dict(),
                }
                torch.save(save_dict, os.path.join(args.logdir, "ckpts", 'best_ckpt.pth'))
                cur_psnr[0] = psnr_mu.avg
                msg = 'Best epoch:' + str(epoch) + '\n'
                print(msg); f.write(msg+"\n")
                msg = 'Testing set: Average PSNR: {:.4f}, PSNR_mu: {:.4f}, SSIM_l: {:.4f}, SSIM_mu: {:.4f}\n'.format(
                    psnr_l.avg,
                    psnr_mu.avg,
                    ssim_l.avg,
                    ssim_mu.avg
                )
                print(msg); f.write(msg+"\n")

def main():
    # settings
    args = get_args()
    # random seed
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process and not os.path.exists(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(os.path.join(args.logdir, "ckpts"), exist_ok=True)
    # model architectures
    if accelerator.is_main_process: print(f'Selected network: {args.model}')
    if args.model == "SCTNet":
        upscale = 4
        window_size = 8
        height = (256 // upscale // window_size + 1) * window_size
        width = (256 // upscale // window_size + 1) * window_size
        model = SCTNet(upscale=2, img_size=(height, width), in_chans=18,
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    elif args.model == "HDR-Transformer":
        model = HDRTransformer(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6)
    elif args.model == "SAFNet":
        model = SAFNet()
        args.loss_func = 3
    else:
        raise ValueError(f"Don't support {args.model}")
    # dataset and dataloader
    if args.model == "SAFNet":
        if args.data_name.lower() in ("sig17", "sct"):
            train_dataset = SAFNet_SIG17_Training_Dataset(root_dir=args.dataset_dir, sub_set="Training", is_training=True)
        elif args.data_name in ("sig17-cache", "sct-cache", ):
            train_dataset = SIG17_Training_Dataset_Cache(root_dir=args.dataset_dir, sub_set="Training", patch_size=512, cache=args.cache)
        elif args.data_name in ("sct-cha-cache", ):
            train_dataset1 = SIG17_Training_Dataset_Cache(root_dir=args.dataset_dir, sub_set="Training", patch_size=512, cache=args.cache)
            train_dataset2 = Challenge123_Training_Dataset_Cache(root_dir=args.dataset_dir2, sub_set="Training", patch_size=512, cache=args.cache)
            train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        elif args.data_name == "challenge123-cache":
            train_dataset = Challenge123_Training_Dataset_Cache(root_dir=args.dataset_dir, sub_set="Training", patch_size=512, cache=args.cache)
        elif args.data_name == "s2r-hdr":
            train_dataset = RenderFlowDataset(root_dir=args.dataset_dir, nframes=3, nexps=3, cache=args.cache, task="train", scale=args.scale, scale_value=args.scale_value, patch_size=512)
    else:
        if args.data_name.lower() in ("sig17", "sct"):
            train_dataset = SIG17_Training_Dataset(root_dir=args.dataset_dir, sub_set=args.sub_set, is_training=True)
        elif args.data_name in ("sig17-cache", "sct-cache", "hdri-cache"):
            train_dataset = SIG17_Training_Dataset_Cache(root_dir=args.dataset_dir, sub_set="Training", patch_size=128, cache=args.cache)
        elif args.data_name.lower() in ("sct-cha-cache"):
            train_dataset1 = SIG17_Training_Dataset_Cache(root_dir=args.dataset_dir, sub_set="Training", patch_size=128, cache=args.cache)
            train_dataset2 = Challenge123_Training_Dataset_Cache(root_dir=args.dataset_dir2, sub_set="Training", patch_size=128, cache=args.cache)
            train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        elif args.data_name == "challenge123-cache":
            train_dataset = Challenge123_Training_Dataset_Cache(root_dir=args.dataset_dir, sub_set="Training", patch_size=128, cache=args.cache)
        elif args.data_name == "s2r-hdr":
            train_dataset = RenderFlowDataset(root_dir=args.dataset_dir, nframes=3, nexps=3, cache=args.cache, task="train", scale=args.scale, scale_value=args.scale_value)
        else:
            raise ValueError(f"Don't support dataset name: {args.data_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # init
    if args.init_weights:
        init_parameters(model)
    # loss
    loss_dict = {
        0: L1MuLoss,
        1: JointReconPerceptualLoss,
        2: JointReconGammaPerceptualLoss,
        3: SAFNetLoss,
    }
    criterion = loss_dict[args.loss_func]().to(accelerator.device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(len(train_loader) * args.epochs), eta_min=args.lr_min, last_epoch=-1, verbose=False) # 20

    if accelerator.is_main_process:
        with open(os.path.join(args.logdir, 'train_log.txt'), 'w+') as f:
            f.write(f"Training begin time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            for k, v in vars(args).items():
                print(k, ":", v); f.write(f"{k}: {v}\n")
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['state_dict'].items()})
            if args.start_epoch != 1:
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            if accelerator.is_main_process:
                with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
                    msg = "===> Loading checkpoint from: {}".format(args.resume)
                    print(msg); f.write(msg+"\n")
                    msg = "===> Loaded checkpoint: epoch {}".format(checkpoint['epoch'])
                    print(msg); f.write(msg+"\n")
        else:
            if accelerator.is_main_process:
                with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
                    msg = "==> No checkpoint is founded at {}.".format(args.resume)
                    print(msg); f.write(msg+"\n")
    model.to(accelerator.device)
    if args.debug:
        args.cache = False
    if args.val:
        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)
    else:
        model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, train_loader)
    
    if accelerator.is_main_process:
        msg = f'''===> Start training {args.model}

            Train Dataset dir1:{args.dataset_dir}
            Train Dataset dir2:{args.dataset_dir2}
            Subset:          {args.sub_set}
            Test Dataset dir:{args.test_dataset_dir}
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Loss function:   {args.loss_func}
            Learning rate:   {args.lr}
            Training size:   {len(train_loader.dataset)}
            Dataloader size: {len(train_loader)}
            Device:          {device.type}({torch.cuda.device_count()})
            '''
        with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
            print(msg); f.write(msg+"\n")
    
    cur_psnr = [-1.0]
    for epoch in range(args.start_epoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion)
        accelerator.wait_for_everyone()
        if (epoch % args.test_interval == 0 or epoch == args.epochs or epoch == args.start_epoch):
            test(args, model, device, optimizer, scheduler, epoch, cur_psnr)
        accelerator.wait_for_everyone()
        if args.debug:
            print("Debug Successful!!")
            break


if __name__ == '__main__':
    main()