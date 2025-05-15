# -*- coding:utf-8 -*-
import os
import time
import argparse
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import pyiqa
from torch.utils.data import DataLoader
from dataset.dataset_nogt import SAFNet_Test_NoGT_Dataset, TestNoGTDataset, TTA_Img_Dataset_NoGT
from dataset.dataset_sig17 import SIG17_Test_Dataset, Challenge123_Test_Dataset, TTA_Img_Dataset
from dataset.dataset_safnet import SAFNet_SIG17_Training_Dataset, SAFNet_Challenge123_Val_Dataset
from models.loss import L1MuLoss, JointReconPerceptualLoss, JointReconGammaPerceptualLoss, SAFNetLoss
from models.hdr_transformer import HDRTransformer
from models.SCTNet import SCTNet, inject_trainable_adapter_transformer
from models.SAFNet import SAFNet, inject_trainable_adapter_cnn
from utils.utils import *
from utils.aug import ExpAug, WBAug, PermAug, FlipAug
from copy import deepcopy



def create_ema_model(model):
    ema_model = deepcopy(model) # get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def update_ema_variables(ema_model, model, alpha_model, alpha_prompt, iteration=None):
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
            if "prompt" in name:
                ema_param.data[:] = alpha_prompt * ema_param[:].data[:] + (1 - alpha_prompt) * param[:].data[:]
            else:
                ema_param.data[:] = alpha_model * ema_param[:].data[:] + (1 - alpha_model) * param[:].data[:]
    return ema_model



def get_args():
    parser = argparse.ArgumentParser(description='ImageHDR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='HDR-Transformer', choices=['HDR-Transformer', 'SCTNet', "SAFNet"])
    parser.add_argument("--data_name", type=str, default="sct", choices=["sig17", "sct", "s2r-hdr", "challenge123-cache", "sig17-cache", "sct-cache"],
                        help='dataset directory'),
    parser.add_argument("--dataset_dir", type=str, default='../../datasets/ImageHDR/sct',
                        help='dataset directory'),
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
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='training batch size (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=24, metavar='N',
                        help='testing batch size (default: 24)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
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
    parser.add_argument('--save_results', action='store_true', default=False, help='debug')
    parser.add_argument('--no_gt', action='store_true', default=False, help='debug')
    parser.add_argument('--no_gt_metric', action='store_true', default=False, help='metric like musiq?')
    parser.add_argument('--adapter', action='store_true', default=False, help='vida?')
    parser.add_argument('--adaptive_scale', action='store_true', default=False, help='vida?')
    parser.add_argument('--r1', default=1, type=int, help='sim')
    parser.add_argument('--r2', default=64, type=int, help='real')
    parser.add_argument('--scale1', default=1.0, type=float, help='sim')
    parser.add_argument('--scale2', default=1.0, type=float, help='real')

    parser.add_argument('--prompt_lr', default=1e-4, type=float, help='scale value')
    parser.add_argument('--adapter_lr', default=1e-4, type=float, help='scale value')
    parser.add_argument('--model_lr', default=1e-4, type=float, help='scale value')
    parser.add_argument('--adapter_lr_scale', default=0, type=float, help='adapter_lr = model_lr*scale : [1, 0.5, 0.2, 0.1]')
    parser.add_argument('--prompt_ema_rate', default=0.999, type=float, help='scale value')
    parser.add_argument('--model_ema_rate', default=0.999, type=float, help='scale value')
    parser.add_argument('--tta_img_num_patches', default=100, type=int, help='scale value')
    parser.add_argument('--tta_data_random_crop', default=True, type=bool, help='scale value')
    parser.add_argument('--tta_aug_type', default=0, type=int, choices=[0, 1, 2], help='0 means no aug')

    parser.add_argument('--global_res_dir', default="xxx/experiments/res", type=str, help='scale value')


    return parser.parse_args()


def set_scale(update_model, scale1, scale2, device='cuda'):
    for name, module in update_model.named_modules():
        if hasattr(module, 'scale1'):
            module.scale1 = scale1
        if hasattr(module, 'scale2'):
            module.scale2 = scale2


def train(args, model, device, train_loader, optimizer, epoch, criterion, anchor=None,
                    ema_model=None,
                    anchor_model=None,
                    dynamic_ema=False,):
    # loading tta test data (data for tta test)
    if args.model == "SAFNet":
        if "challenge" in args.test_dataset_dir.lower():
            dataset_test = SAFNet_Challenge123_Val_Dataset(args.test_dataset_dir, "Test")
        else:
            dataset_test = SAFNet_SIG17_Training_Dataset(args.test_dataset_dir, "Test", is_training=False)  # dataset modified
        test_datasets = DataLoader(dataset=dataset_test, batch_size=1, num_workers=1, shuffle=False)
        # safnet: full res testing / training 512
    else:
        if "challenge" in args.test_dataset_dir.lower():
            test_datasets = Challenge123_Test_Dataset(args.test_dataset_dir, args.patch_size) 
        else:
            test_datasets = SIG17_Test_Dataset(args.test_dataset_dir, args.patch_size) 
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    msg = f'''===> Start TTA {args.model}

        TTA Dataset dir:{args.test_dataset_dir}
        Subset:          {args.sub_set}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Loss function:   {args.loss_func}
        Learning rate:   {args.model_lr}
        PEFT LR:         {args.model_lr}
        Device:          {device.type}({torch.cuda.device_count()})
        '''
    with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
        print(msg); f.write(msg+"\n")

    # model.train()
    anchor_model.eval()
    param_list = []
    param_list_1 = []

    # print(">>>>>>>>> model details: ")
    # print(model)

    for name, param in model.named_parameters():
        if param.requires_grad and "adapter_" in name:
            param_list.append(param)
            print(name)
        elif param.requires_grad and "adapter_" not in name:
            param_list_1.append(param)
            print(name)
        else:
            param.requires_grad=False
    
    optimizer = torch.optim.Adam([{"params": param_list, "lr": args.adapter_lr if (args.adapter_lr_scale == 0) else args.model_lr * args.adapter_lr_scale},
                                  {"params": param_list_1, "lr": args.model_lr}],
                                 lr=1e-5, betas=(0.9, 0.999), eps=1e-08) #Batchsize=1 now, was 8 during cityscapes training

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()


    with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
        for idx, img_dataset in enumerate(test_datasets):
            if args.model != "SAFNet":
                tta_test_loader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=4, shuffle=False)
                tta_img_dataset = TTA_Img_Dataset(scene=img_dataset.scene, ldr_path=img_dataset.ldr_path, label_path=img_dataset.label_path, exposure_path=img_dataset.exposure_path, 
                                    patch_size=(128 if args.model != "SAFNet" else 512), num_patches=args.tta_img_num_patches, random_crop=args.tta_data_random_crop)
                tta_train_loader = DataLoader(dataset=tta_img_dataset, batch_size=1, num_workers=4, shuffle=False)
            else:
                tta_test_loader = None  # the img_dataset it self is the batch_data
                tta_img_dataset = TTA_Img_Dataset(scene=img_dataset['scene'][0], 
                                                  ldr_path=[img_dataset['ldr_path'][0][0],img_dataset['ldr_path'][1][0],img_dataset['ldr_path'][2][0]], 
                                                  label_path=img_dataset['label_path'][0], 
                                                  exposure_path=img_dataset['exposure_path'][0], 
                                                  patch_size=512, num_patches=args.tta_img_num_patches, random_crop=args.tta_data_random_crop)
                tta_train_loader = DataLoader(dataset=tta_img_dataset, batch_size=1, num_workers=4, shuffle=False)


            for batch_idx, batch_data in enumerate(tta_train_loader):
                data_time.update(time.time() - end)

                model.eval()
                ema_model.eval()
                anchor_model.eval()


                batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                                    batch_data['input2'].to(device)

                with torch.no_grad():
                    if args.loss_func == 3:
                        ps_label_m, ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()) 
                    else:
                        ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()) 

                    if args.tta_aug_type > 0: 
                        ps_label_aug = [ps_label]
                        if args.tta_aug_type == 1:
                            aug_list = [FlipAug(0), FlipAug(1), PermAug(), PermAug()]
                        elif args.tta_aug_type == 2:
                            aug_list = [FlipAug(0), FlipAug(1), PermAug(), 
                                        ExpAug(0.6, batch_data['exp'][0].to(device)), 
                                        ExpAug(-0.5, batch_data['exp'][0].to(device)), 
                                        ExpAug(-1, batch_data['exp'][0].to(device)), 
                                        ExpAug(0.3, batch_data['exp'][0].to(device)), 
                                        WBAug([1.0, 0.9, 1.1], batch_data['exp'][0].to(device)), 
                                        WBAug([0.9, 0.8, 1.2], batch_data['exp'][0].to(device))]
                        else:
                            raise NotImplementedError()
                        for id_aug in range(len(aug_list)):
                            if args.loss_func == 3:
                                _, outputs_ = ema_model(aug_list[id_aug].transform(batch_ldr0, 0).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr1, 1).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr2, 2).contiguous())
                            else:
                                outputs_ = ema_model(aug_list[id_aug].transform(batch_ldr0, 0).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr1, 1).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr2, 2).contiguous())
                            outputs_ = aug_list[id_aug].transform_back(outputs_)
                            ps_label_aug.append(outputs_)
                        ps_label = torch.stack(ps_label_aug)
                        variance = torch.var(ps_label)
                        uncertainty = torch.mean(variance) * 0.1
                        ps_label = torch.mean(ps_label, dim=0)
                        if args.adaptive_scale:
                            set_scale(model, 1 - uncertainty, 1 + uncertainty)
                            set_scale(ema_model, 1 - uncertainty, 1 + uncertainty)
                            if args.loss_func == 3:
                                _, ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()) 
                            else:
                                ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())

                if args.loss_func == 3:
                    pred_m, pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    loss = criterion(pred, ps_label, pred_m)
                elif args.loss_func == 2:
                    pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    exp = batch_data['exp'].to(device)
                    loss = criterion(pred, ps_label, exp)
                else:
                    pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    loss = criterion(pred, ps_label)
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_model=args.model_ema_rate, alpha_prompt=args.prompt_ema_rate)
                if True:
                    for npp, p in model.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape)<0.01).float().cuda()  # new hyperparam here
                            with torch.no_grad():
                                p.data = anchor[npp] * mask + p * (1.-mask)

                batch_time.update(time.time() - end)
                end = time.time()
                if batch_idx % args.log_interval == 0:
                    msg = '{} TTA Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t' \
                        'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'\
                        'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        batch_idx * args.batch_size,
                        len(tta_train_loader.dataset),
                        100. * batch_idx * args.batch_size / len(tta_train_loader.dataset),
                        loss.item(),
                        batch_time=batch_time,
                        data_time=data_time
                    )
                    print(msg);f.write(msg + '\n')
                    if args.debug:
                        break

            print('tta testing this img')
            with torch.no_grad():
                if args.model != "SAFNet":
                    myscene = '0'
                    for scene, batch_data in tqdm(tta_test_loader, total=len(tta_test_loader)):
                        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), \
                                                        batch_data['input1'].to(device), \
                                                        batch_data['input2'].to(device)
                        myscene = scene
                        output = ema_model(batch_ldr0, batch_ldr1, batch_ldr2).clip(0.0, 1.0)
                        
                        for i in range(output.shape[0]):
                            img_dataset.update_result(output[i].detach().cpu().numpy().astype(np.float32))
                    pred_img, label = img_dataset.rebuild_result()
                    scene = myscene
                else:
                    batch_ldr0, batch_ldr1, batch_ldr2 = img_dataset['input0'].to(device), \
                                                        img_dataset['input1'].to(device), \
                                                        img_dataset['input2'].to(device)
                    label = img_dataset['label'].to(device) 
                    scene = img_dataset['scene'][0]
                    _, pred_img = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    pred_img = pred_img.detach().cpu().numpy()[0]
                    label = label.detach().cpu().numpy()[0]

                pred_hdr = pred_img.copy()
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
                print(f'tta this img result:psnr_l {scene_psnr_l} ; ssim_l {scene_ssim_l}; psnr_mu {scene_psnr_mu}; ssim_mu {scene_ssim_mu}')
                if args.save_results:
                    pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1]
                    # pred_img_mu = range_compressor(pred_hdr).transpose(1, 2, 0)
                    img_save_path = os.path.join(args.logdir, 'img_results', f'ep_{epoch}')
                    os.makedirs(img_save_path, exist_ok=True)
                    save_hdr(os.path.join(img_save_path, f'{idx}_{scene}_{args.model}_ep{epoch}.hdr'), pred_hdr)
                    cv2.imwrite(os.path.join(img_save_path, f'{idx}_{scene}_{args.model}_ep{epoch}.png'), (pred_img_mu).astype(np.uint8))
        msg = '!!!!![TTA round DONE]!!!!!\n->\n==Testing==\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}'.format(
            psnr_l.avg,
            psnr_mu.avg,
            ssim_l.avg,
            ssim_mu.avg
        )
        print(msg); f.write(msg+"\n")

    os.makedirs(args.global_res_dir, exist_ok=True)
    if args.epochs > 1:
        res_fn = "res_multi_epoch.txt"
    else:
        res_fn = "res.txt"
    with open(os.path.join(args.global_res_dir, res_fn), 'a') as f:
        msg = f'==TTA==\t epoch[{epoch}]\t' + 'PSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}\t'.format(
            psnr_l.avg,
            psnr_mu.avg,
            ssim_l.avg,
            ssim_mu.avg
        )
        msg = msg + "dir: " + args.logdir
        print(msg); f.write(msg+"\n")
    
    # capture metrics
    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    if epoch % args.ckpt_interval == 0 or (epoch == args.epochs - 1):
        torch.save(save_dict, os.path.join(args.logdir, "ckpts", f'epoch_{epoch}.pth'))
    torch.save(save_dict, os.path.join(args.logdir, "ckpts", f'last_ckpt.pth'))


def train_nogt(args, model, device, train_loader, optimizer, epoch, criterion, anchor=None,
                    ema_model=None,
                    anchor_model=None,
                    dynamic_ema=False,):
    # loading tta test data (data for tta test)
    if args.model == "SAFNet":
        dataset_test = SAFNet_Test_NoGT_Dataset(args.test_dataset_dir)
        test_datasets = DataLoader(dataset=dataset_test, batch_size=1, num_workers=1, shuffle=False)
        # safnet: full res testing / training 512
    else:
        test_datasets = TestNoGTDataset(args.test_dataset_dir, args.patch_size) 
    
    if args.no_gt_metric:
        musiq_l = AverageMeter()
        musiq_metric = pyiqa.create_metric('musiq').cuda()
    msg = f'''===> Start TTA {args.model}

        TTA Dataset dir:{args.test_dataset_dir}
        Subset:          {args.sub_set}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Loss function:   {args.loss_func}
        Learning rate:   {args.model_lr}
        PEFT LR:         {args.model_lr}
        Device:          {device.type}({torch.cuda.device_count()})
        '''
    with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
        print(msg); f.write(msg+"\n")

    # model.train()
    anchor_model.eval()
    param_list = []
    param_list_1 = []

    for name, param in model.named_parameters():
        if param.requires_grad and "prompt" in name:
            param_list.append(param)
            print(name)
        elif param.requires_grad and "prompt" not in name:
            param_list_1.append(param)
            print(name)
        else:
            param.requires_grad=False
    
    optimizer = torch.optim.Adam([{"params": param_list, "lr": args.prompt_lr},
                                  {"params": param_list_1, "lr": args.model_lr}],
                                 lr=1e-5, betas=(0.9, 0.999), eps=1e-08) #Batchsize=1 now, was 8 during cityscapes training

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()


    with open(os.path.join(args.logdir, 'train_log.txt'), 'a') as f:
        for idx, img_dataset in enumerate(test_datasets):
            if args.model != "SAFNet":
                tta_test_loader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=4, shuffle=False)
                tta_img_dataset = TTA_Img_Dataset_NoGT(scene=img_dataset.scene, ldr_path=img_dataset.ldr_path, exposure_path=img_dataset.exposure_path, 
                                    patch_size=(128 if args.model != "SAFNet" else 512), num_patches=args.tta_img_num_patches, random_crop=args.tta_data_random_crop)
                tta_train_loader = DataLoader(dataset=tta_img_dataset, batch_size=1, num_workers=4, shuffle=False)
            else:
                tta_test_loader = None  # the img_dataset it self is the batch_data
                tta_img_dataset = TTA_Img_Dataset_NoGT(scene=img_dataset['scene'][0], 
                                                  ldr_path=[img_dataset['ldr_path'][0][0],img_dataset['ldr_path'][1][0],img_dataset['ldr_path'][2][0]], 
                                                  exposure_path=img_dataset['exposure_path'][0], 
                                                  patch_size=512, num_patches=args.tta_img_num_patches, random_crop=args.tta_data_random_crop)
                tta_train_loader = DataLoader(dataset=tta_img_dataset, batch_size=1, num_workers=4, shuffle=False)

            for batch_idx, batch_data in enumerate(tta_train_loader):
                data_time.update(time.time() - end)

                model.eval()
                ema_model.eval()
                anchor_model.eval()


                batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                                    batch_data['input2'].to(device)

                with torch.no_grad():
                    if args.loss_func == 3:
                        ps_label_m, ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()) 
                    else:
                        ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()) 
                    if args.tta_aug_type > 0: 
                        ps_label_aug = [ps_label]
                        if args.tta_aug_type == 1:
                            aug_list = [FlipAug(0), FlipAug(1), PermAug(), PermAug()]
                        elif args.tta_aug_type == 2:
                            aug_list = [FlipAug(0), FlipAug(1), PermAug(), 
                                        ExpAug(0.6, batch_data['exp'][0].to(device)), 
                                        ExpAug(-0.5, batch_data['exp'][0].to(device)), 
                                        ExpAug(-1, batch_data['exp'][0].to(device)), 
                                        ExpAug(0.3, batch_data['exp'][0].to(device)), 
                                        WBAug([1.0, 0.9, 1.1], batch_data['exp'][0].to(device)), 
                                        WBAug([0.9, 0.8, 1.2], batch_data['exp'][0].to(device))]
                        else:
                            raise NotImplementedError()
                        for id_aug in range(len(aug_list)):
                            if args.loss_func == 3:
                                _, outputs_ = ema_model(aug_list[id_aug].transform(batch_ldr0, 0).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr1, 1).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr2, 2).contiguous())
                            else:
                                outputs_ = ema_model(aug_list[id_aug].transform(batch_ldr0, 0).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr1, 1).contiguous(), 
                                                    aug_list[id_aug].transform(batch_ldr2, 2).contiguous())
                            outputs_ = aug_list[id_aug].transform_back(outputs_)
                            ps_label_aug.append(outputs_)
                        ps_label = torch.stack(ps_label_aug)
                        variance = torch.var(ps_label)
                        uncertainty = torch.mean(variance) * 0.05
                        ps_label = torch.mean(ps_label, dim=0)
                        if args.adaptive_scale:
                            set_scale(model, 1 - uncertainty, 1 + uncertainty)
                            set_scale(ema_model, 1 - uncertainty, 1 + uncertainty)
                            if args.loss_func == 3:
                                _, ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous()) 
                            else:
                                ps_label = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())

                if args.loss_func == 3:
                    pred_m, pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    loss = criterion(pred, ps_label, pred_m)
                elif args.loss_func == 2:
                    pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    exp = batch_data['exp'].to(device)
                    loss = criterion(pred, ps_label, exp)
                else:
                    pred = model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    loss = criterion(pred, ps_label)
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_model=args.model_ema_rate, alpha_prompt=args.prompt_ema_rate)
                if True:
                    for npp, p in model.named_parameters():
                        if npp in ['weight', 'bias'] and p.requires_grad:
                            mask = (torch.rand(p.shape)<0.01).float().cuda()  # new hyperparam here
                            with torch.no_grad():
                                p.data = anchor[npp] * mask + p * (1.-mask)

                batch_time.update(time.time() - end)
                end = time.time()
                if batch_idx % args.log_interval == 0:
                    msg = '{} TTA Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t' \
                        'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'\
                        'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        batch_idx * args.batch_size,
                        len(tta_train_loader.dataset),
                        100. * batch_idx * args.batch_size / len(tta_train_loader.dataset),
                        loss.item(),
                        batch_time=batch_time,
                        data_time=data_time
                    )
                    print(msg);f.write(msg + '\n')
                    if args.debug:
                        break

            # testing data -> get reported tta result
            print('tta testing this img')
            with torch.no_grad():
                if args.model != "SAFNet":
                    for scene, batch_data in tqdm(tta_test_loader, total=len(tta_test_loader)):
                        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), \
                                                        batch_data['input1'].to(device), \
                                                        batch_data['input2'].to(device)
                        output = ema_model(batch_ldr0, batch_ldr1, batch_ldr2).clip(0.0, 1.0)
                        # ! should the output be simple ema_teacher output, or robust output with unc
                        scene = batch_data['scene'][0]
                        for i in range(output.shape[0]):
                            img_dataset.update_result(output[i].detach().cpu().numpy().astype(np.float32))
                    pred_img, _ = img_dataset.rebuild_result()
                else:
                    batch_ldr0, batch_ldr1, batch_ldr2 = img_dataset['input0'].to(device), \
                                                        img_dataset['input1'].to(device), \
                                                        img_dataset['input2'].to(device)
                    scene = img_dataset['scene'][0]
                    _, pred_img = ema_model(batch_ldr0.contiguous(), batch_ldr1.contiguous(), batch_ldr2.contiguous())
                    pred_img = pred_img.detach().cpu().numpy()[0]
                pred_hdr = pred_img.copy()
                pred_img_mu = range_compressor(pred_img)
                pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
                if args.save_results:
                    pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1]
                    # pred_img_mu = range_compressor(pred_hdr).transpose(1, 2, 0)
                    img_save_path = os.path.join(args.logdir, 'img_results', f'ep_{epoch}')
                    os.makedirs(img_save_path, exist_ok=True)
                    save_hdr(os.path.join(img_save_path, f'{idx}_{scene}_{args.model}_ep{epoch}.hdr'), pred_hdr)
                    cv2.imwrite(os.path.join(img_save_path, f'{idx}_{scene}_{args.model}_ep{epoch}.png'), (pred_img_mu).astype(np.uint8))

                    if args.no_gt_metric:
                        musiq_mu = musiq_metric(os.path.join(img_save_path, f'{idx}_{scene}_{args.model}_ep{epoch}.png'))
                        musiq_l.update(musiq_mu.item())

    print('TTA this round done')

    if args.no_gt_metric:
        os.makedirs(args.global_res_dir, exist_ok=True)
        if args.epochs > 1:
            res_fn = "res_multi_epoch.txt"
        else:
            res_fn = "res.txt"
        with open(os.path.join(args.global_res_dir, res_fn), 'a') as f:
            msg = f'==TTA==\t epoch[{epoch}]\t' + 'musiq_mu: {:.4f}\t'.format(
                musiq_l.avg
            )
            msg = msg + "dir: " + args.logdir
            print(msg); f.write(msg+"\n")

    # capture metrics
    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    if epoch % args.ckpt_interval == 0 or (epoch == args.epochs - 1):
        torch.save(save_dict, os.path.join(args.logdir, "ckpts", f'epoch_{epoch}.pth'))
    torch.save(save_dict, os.path.join(args.logdir, "ckpts", f'last_ckpt.pth'))



def test_single_img(model, img_dataset, device, batch_size=1):
    dataloader = DataLoader(dataset=img_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    # model.eval()
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

def test(args, model, device, optimizer, epoch, cur_psnr, **kwargs):
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
        if args.model == "SAFNet":
            batch_ldr0, batch_ldr1, batch_ldr2 = img_dataset['input0'].to(device), img_dataset['input1'].to(device), \
                img_dataset['input2'].to(device)
            label = img_dataset['label'].to(device) 
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
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
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
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
        os.makedirs(os.path.join(args.logdir, "ckpts"), exist_ok=True)
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # model architectures
    print(f'Selected network: {args.model}')
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
    cur_psnr = [-1.0]
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
    criterion = loss_dict[args.loss_func]().to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=, eta_min=0, last_epoch=-1, verbose=False)
    
    with open(os.path.join(args.logdir, 'train_log.txt'), 'w+') as f:
        f.write(f"Training begin time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        for k, v in vars(args).items():
            print(k, ":", v); f.write(f"{k}: {v}\n")
        if args.resume:
            if os.path.isfile(args.resume):
                msg = "===> Loading checkpoint from: {}".format(args.resume)
                print(msg); f.write(msg+"\n")
                checkpoint = torch.load(args.resume)
                model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['state_dict'].items()})
                # args.start_epoch = checkpoint['epoch']
                # optimizer.load_state_dict(checkpoint['optimizer'])
                msg = "===> Loaded checkpoint: epoch {}".format(checkpoint['epoch'])
                print(msg); f.write(msg+"\n")
            else:
                msg = "==> No checkpoint is founded at {}.".format(args.resume)
                print(msg); f.write(msg+"\n")
        if args.adapter:
            if args.model == "SCTNet":
                inject_trainable_adapter_transformer(model, r1=args.r1, r2=args.r2)
                msg = "Insert adapter ot SCTNet"
                print(msg); f.write(msg+"\n")
            elif args.model == "SAFNet":
                inject_trainable_adapter_cnn(model, r1=args.r1, r2=args.r2)
                msg = "Insert adapter to SAFNet"
                print(msg); f.write(msg+"\n")
            else:
                raise ValueError("Current just support SCTNet and SAFNet adapter")
            set_scale(model, scale1=args.scale1, scale2=args.scale2, device=device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.debug:
        args.cache = False
    else:
        args.cache = True
    
    for epoch in range(args.epochs):  # TTA rounds
        adjust_learning_rate(args, optimizer, epoch)
        anchor = deepcopy(model.state_dict())
        anchor_model = deepcopy(model)
        ema_model = create_ema_model(model)

        if args.no_gt:
            train_nogt(args, model, device, None, optimizer, epoch, criterion, anchor, ema_model, anchor_model)
        else:
            train(args, model, device, None, optimizer, epoch, criterion, anchor, ema_model, anchor_model)

        if args.debug:
            print("Debug Successful!!")
            break



if __name__ == '__main__':
    main()

