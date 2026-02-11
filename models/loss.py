#-*- coding:utf-8 -*-  
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super(JointReconPerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_vgg
        return loss


def gamma_correction(x, gamma=1/2.2, exp=1.0):
    return torch.pow(x * exp[:, None, None, None], gamma).clip(0.0, 1.0)


def weight_3expo_low(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.0
    mask2 = img >= 0.50
    w[mask2] = img[mask2] - 0.5
    w /= 0.5
    return w


def weight_3expo_mid(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 1.0 - img[mask2]
    w /= 0.5
    return w

def weight_3expo_high(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.5 - img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 0.0
    w /= 0.5
    return w


class L1GammaLoss(nn.Module):
    def __init__(self, gamma=1/2.2):
        super(L1GammaLoss, self).__init__()
        self.gamma = gamma
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, pred, label, exp):
        weight_low = weight_3expo_low(label).detach()
        weight_mid = weight_3expo_mid(label).detach()
        weight_high = weight_3expo_high(label).detach()
        
        pred_low = gamma_correction(pred, self.gamma, exp[:, 0])
        pred_mid = gamma_correction(pred, self.gamma, exp[:, 1])
        pred_high = gamma_correction(pred, self.gamma, exp[:, 2])

        label_low = gamma_correction(label, self.gamma, exp[:, 0])
        label_mid = gamma_correction(label, self.gamma, exp[:, 1])
        label_high = gamma_correction(label, self.gamma, exp[:, 2])
        # print(weight_low, weight_mid, weight_high)
        loss_low = self.loss(pred_low, label_low) * weight_low
        loss_mid = self.loss(pred_mid, label_mid) * weight_mid
        loss_high = self.loss(pred_high, label_high) * weight_high
        return ((loss_low + loss_mid + loss_high) / 3).mean()


class JointReconGammaPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000, gamma=1/2.2):
        super(JointReconGammaPerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(False)
        self.gamma = gamma
        self.loss_gamma_recon = L1GammaLoss(self.gamma)

    def forward(self, input, target, exp):
        """
        input: [b, c, h, w]
        target: [b, c, h, w]
        exp: [b, 3]
        """
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss_gamma_recon = self.loss_gamma_recon(input, target, exp)
        # print("loss", exp, loss_recon, loss_gamma_recon, loss_vgg)
        loss = loss_recon + loss_gamma_recon * 0.1 + self.alpha * loss_vgg
        return loss


class Ternary(nn.Module):
    def __init__(self, patch_size=7):
        super(Ternary, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().cuda()

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask
        
    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss


class JointReconCensusLoss(nn.Module):
    def __init__(self, alpha=1.0, mu=5000, kernel_size=7):
        super(JointReconCensusLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.kernel_size = kernel_size
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_census = Ternary(self.kernel_size)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        loss_recon = self.loss_recon(input, target)
        loss_census = self.loss_census(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_census
        return loss

class SAFNetLoss(nn.Module):
    def __init__(self, alpha=0.01, beta=0.1, mu=5000, kernel_size=7):
        super(SAFNetLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.beta = beta
        self.kernel_size = kernel_size
        self.loss_r = JointReconPerceptualLoss(alpha, mu)
        self.loss_m = JointReconCensusLoss(1.0, mu=mu, kernel_size=kernel_size)

    def forward(self, input_r, target, input_m):
        loss_r = self.loss_r(input_r, target)
        loss_m = self.loss_m(input_m, target)
        loss = loss_r + self.beta * loss_m
        return loss


if __name__ == "__main__":
    # loss = JointReconGammaPerceptualLoss()
    # x, y = torch.rand((2, 1, 4, 4)), torch.rand((2, 1, 4, 4))
    # exp = torch.tensor([2, 3])
    # print(loss(x, y, exp))

    loss = SAFNetLoss()
    x, y, z = torch.rand((2, 1, 512, 512)), torch.rand((2, 1, 512, 512)), torch.rand((2, 1, 512, 512))
    print(loss(x, y, z))
