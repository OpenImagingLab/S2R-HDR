import sys
sys.path.append('.')
sys.path.append('..')
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

div_size = 16
div_flow = 20.0

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

def weight_3expo_low_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.0
    mask2 = img >= 0.50
    w[mask2] = img[mask2] - 0.5
    w /= 0.5
    return w

def weight_3expo_mid_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 1.0 - img[mask2]
    w /= 0.5
    return w

def weight_3expo_high_tog17(img):
    w = torch.zeros_like(img)
    mask1 = img < 0.5
    w[mask1] = 0.5 - img[mask1]
    mask2 = img >= 0.5
    w[mask2] = 0.0
    w /= 0.5
    return w

def merge_hdr(ldr_imgs, lin_imgs, mask0, mask2):
    sum_img = torch.zeros_like(ldr_imgs[1])
    sum_w = torch.zeros_like(ldr_imgs[1])
    w_low = weight_3expo_low_tog17(ldr_imgs[1]) * mask0
    w_mid = weight_3expo_mid_tog17(ldr_imgs[1]) + weight_3expo_low_tog17(ldr_imgs[1]) * (1.0 - mask0) + weight_3expo_high_tog17(ldr_imgs[1]) * (1.0 - mask2)
    w_high = weight_3expo_high_tog17(ldr_imgs[1]) * mask2
    w_list = [w_low, w_mid, w_high]
    for i in range(len(ldr_imgs)):
        sum_w += w_list[i]
        sum_img += w_list[i] * lin_imgs[i]
    hdr_img = sum_img / (sum_w + 1e-9)
    return hdr_img


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(6, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(40, 40, 3, 2, 1), 
            convrelu(40, 40, 3, 1, 1)
        )
        
    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = convrelu(126, 120)
        self.conv2 = convrelu(120, 120, groups=3)
        self.conv3 = convrelu(120, 120, groups=3)
        self.conv4 = convrelu(120, 120, groups=3)
        self.conv5 = convrelu(120, 120)
        self.conv6 = deconv(120, 6)

    def forward(self, f0, f1, f2, flow0, flow2, mask0, mask2):
        f0_warp = warp(f0, flow0)
        f2_warp = warp(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2, mask0, mask2], 1)
        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), 3)
        f_out = channel_shuffle(self.conv3(f_out), 3)
        f_out = channel_shuffle(self.conv4(f_out), 3)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)
        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        up_mask0 = resize(mask0, scale_factor=2.0) + f_out[:, 4:5]
        up_mask2 = resize(mask2, scale_factor=2.0) + f_out[:, 5:6]
        return up_flow0, up_flow2, up_mask0, up_mask2


class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias), 
            nn.PReLU(channels)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv0 = nn.Sequential(convrelu(6, 20), convrelu(20, 20))
        self.conv1 = nn.Sequential(convrelu(6+2+2+1+1+3, 40), convrelu(40, 40))
        self.conv2 = nn.Sequential(convrelu(6, 20), convrelu(20, 20))
        self.resblock1 = ResBlock(80, 1)
        self.resblock2 = ResBlock(80, 2)
        self.resblock3 = ResBlock(80, 4)
        self.resblock4 = ResBlock(80, 2)
        self.resblock5 = ResBlock(80, 1)
        self.conv3 = nn.Conv2d(80, 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        
    def forward(self, img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m):
        feat0 = self.conv0(img0_c)
        feat1 = self.conv1(torch.cat([img1_c, flow0 / div_flow, flow2 / div_flow, mask0, mask2, img_hdr_m], 1))
        feat2 = self.conv2(img2_c)
        feat0_warp = warp(feat0, flow0)
        feat2_warp = warp(feat2, flow2)
        feat = torch.cat([feat0_warp, feat1, feat2_warp], 1)
        feat = self.resblock1(feat)
        feat = self.resblock2(feat)
        feat = self.resblock3(feat)
        feat = self.resblock4(feat)
        feat = self.resblock5(feat)
        res = self.conv3(feat)
        img_hdr_r = torch.clamp(img_hdr_m + res, 0, 1)
        return img_hdr_r

def window_paration(x: torch.tensor, H_patch=128, W_patch=128):
    B, C, H, W = x.shape # [b, c, 512, 512]
    H_blocks, W_blocks = H // H_patch, W // W_patch
    # Step 1: 展开
    unfolded = x.unfold(2, H_patch, W_patch).unfold(3, H_patch, W_patch)  # [b, c, 4, 4, 128, 128]
    # Step 2: 重排列维度，准备将块合并到通道维度
    unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()  # [b, 4, 4, c, 128, 128]
    # B, H_blocks, W_blocks, C, H_patch, W_patch = unfolded.shape
    # print("unfolded", unfolded.shape)
    # Step 3: 合并块数和通道数
    reshaped = unfolded.view(B*H_blocks * W_blocks, C, H_patch, W_patch)  # [b, c*4*4, 128, 128]
    return reshaped, B, C, H_blocks, W_blocks

def window_inverse(x: torch.tensor, B, C, H_blocks, W_blocks, H_patch, W_patch):
    reshaped = x.view(B, H_blocks, W_blocks, C, H_patch, W_patch)  # [b, 4, 4, c, 128, 128]
    reshaped = reshaped.permute(0, 3, 1, 2, 4, 5).contiguous()  # [b, c, 4, 4, 128, 128]
    fold_input = reshaped.view(B * C, H_blocks * W_blocks, H_patch * W_patch).permute(0, 2, 1)
    output = F.fold(fold_input, output_size=(H_blocks*H_patch, W_blocks*W_patch),
                    kernel_size=(H_patch, W_patch),
                    stride=(H_patch, W_patch)).view(B, C, H_blocks*H_patch, W_blocks*W_patch)
    return output


class SAFNet(nn.Module):
    def __init__(self):
        super(SAFNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refinenet = RefineNet()

    def forward_flow_mask(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (int(div_size * np.ceil(h * scale_factor / div_size)), int(div_size * np.ceil(w * scale_factor / div_size)))

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_flow2_5 = torch.zeros_like(f1_4[:, 0:2, :, :])
        up_mask0_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_mask2_5 = torch.zeros_like(f1_4[:, 0:1, :, :])
        up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4 = self.decoder(f0_4, f1_4, f2_4, up_flow0_5, up_flow2_5, up_mask0_5, up_mask2_5)
        up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3 = self.decoder(f0_3, f1_3, f2_3, up_flow0_4, up_flow2_4, up_mask0_4, up_mask2_4)
        up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2 = self.decoder(f0_2, f1_2, f2_2, up_flow0_3, up_flow2_3, up_mask0_3, up_mask2_3)
        up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1 = self.decoder(f0_1, f1_1, f2_1, up_flow0_2, up_flow2_2, up_mask0_2, up_mask2_2)

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]
            up_flow0_1 = F.interpolate(up_flow0_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow0_1[:, 0, :, :] *= scale_w
            up_flow0_1[:, 1, :, :] *= scale_h
            up_flow2_1 = F.interpolate(up_flow2_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow2_1[:, 0, :, :] *= scale_w
            up_flow2_1[:, 1, :, :] *= scale_h
            up_mask0_1 = F.interpolate(up_mask0_1, size=org_size, mode='bilinear', align_corners=False)
            up_mask2_1 = F.interpolate(up_mask2_1, size=org_size, mode='bilinear', align_corners=False)

        up_mask0_1 = torch.sigmoid(up_mask0_1)
        up_mask2_1 = torch.sigmoid(up_mask2_1)

        return up_flow0_1, up_flow2_1, up_mask0_1, up_mask2_1
    
    def forward(self, img0_c, img1_c, img2_c, scale_factor=0.5, refine=True):
        # imgx_c[:, 0:3] linear domain, imgx_c[:, 3:6] ldr domain
        flow0, flow2, mask0, mask2 = self.forward_flow_mask(img0_c, img1_c, img2_c, scale_factor=scale_factor)

        img0_c_warp = warp(img0_c, flow0)
        img2_c_warp = warp(img2_c, flow2)
        img_hdr_m = merge_hdr(
            [img0_c_warp[:, 3:6, :, :], img1_c[:, 3:6, :, :], img2_c_warp[:, 3:6, :, :]], 
            [img0_c_warp[:, 0:3, :, :], img1_c[:, 0:3, :, :], img2_c_warp[:, 0:3, :, :]], 
            mask0, mask2
            )
        
        if refine == True:
            if self.training:
                H_patch = 128
                W_patch = 128
    
                img0_c, B, C, H_blocks, W_blocks = window_paration(img0_c, H_patch, W_patch)
                img1_c, B, C, H_blocks, W_blocks = window_paration(img1_c, H_patch, W_patch)
                img2_c, B, C, H_blocks, W_blocks = window_paration(img2_c, H_patch, W_patch)
                flow0, B, C, H_blocks, W_blocks = window_paration(flow0, H_patch, W_patch)
                flow2, B, C, H_blocks, W_blocks = window_paration(flow2, H_patch, W_patch)
                mask0, B, C, H_blocks, W_blocks = window_paration(mask0, H_patch, W_patch)
                mask2, B, C, H_blocks, W_blocks = window_paration(mask2, H_patch, W_patch)
                img_hdr_m_, B, C, H_blocks, W_blocks = window_paration(img_hdr_m, H_patch, W_patch)
                
                img_hdr_r = self.refinenet(img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m_)

                img_hdr_r = window_inverse(img_hdr_r, B, C, H_blocks, W_blocks, H_patch, W_patch)
                return img_hdr_m, img_hdr_r
            else:
                img_hdr_r = self.refinenet(img0_c, img1_c, img2_c, flow0, flow2, mask0, mask2, img_hdr_m)
                return img_hdr_m, img_hdr_r
        else:
            return img_hdr_m


class InjectedConv1x1(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding=1, dilation=1, groups=1, bias=False, r1=1, r2=128):
        super().__init__()

        self.conv_adapter = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.adapter_down = nn.Conv2d(in_features, r1, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=1, bias=False)
        self.adapter_up = nn.Conv2d(r1, out_features, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=1, bias=False)
        self.adapter_down2 = nn.Conv2d(in_features, r2, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=1, bias=False)
        self.adapter_up2 = nn.Conv2d(r2, out_features, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=1, bias=False)
        self.scale1 = 1.0
        self.scale2 = 1.0

        nn.init.normal_(self.adapter_down.weight, std=1 / r1**2)
        nn.init.zeros_(self.adapter_up.weight)

        nn.init.normal_(self.adapter_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.adapter_up2.weight)

    def forward(self, input):
        if isinstance(self.scale1, (int, float)):
            return self.conv_adapter(input) + \
                self.adapter_up(self.adapter_down(input)) * self.scale1 + \
                    self.adapter_up2(self.adapter_down2(input)) * self.scale2
        device = input.device
        return self.conv_adapter(input) + \
                self.adapter_up(self.adapter_down(input)) * self.scale1.to(device) + \
                    self.adapter_up2(self.adapter_down2(input)) * self.scale2.to(device)

def inject_trainable_adapter_cnn(
    model: nn.Module,
    target_replace_module: List[str] = ["ResBlock", "Sequential" ],
    r1: int = 1,
    r2: int = 64,
):
    """
    inject adapter into model, and returns adapter parameter groups.
    """

    require_grad_params = []
    names = []

    for father_name, _module in model.named_modules():
        if _module.__class__.__name__ in target_replace_module:

            for name, _child_module in _module.named_modules():
                # print(father_name, _module.__class__.__name__, _child_module.__class__.__name__, name)
                if (_module.__class__.__name__ == "ResBlock" and _child_module.__class__.__name__ == "Conv2d" and name == "conv2") or \
                    (_module.__class__.__name__ == "Sequential" and _child_module.__class__.__name__ == "Conv2d" and name == "0" and \
                        "decoder" in father_name):

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = InjectedConv1x1(
                        _child_module.in_channels,
                        _child_module.out_channels,
                        _child_module.kernel_size,
                        _child_module.padding,
                        _child_module.dilation,
                        _child_module.groups,
                        bias=_child_module.bias is not None,
                        r1=r1,
                        r2=r2,
                    )
                    _tmp.conv_adapter.weight = weight
                    if bias is not None:
                        _tmp.conv_adapter.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[name].adapter_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].adapter_down.parameters())
                    )
                    _module._modules[name].adapter_up.weight.requires_grad = True
                    _module._modules[name].adapter_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[name].adapter_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].adapter_down2.parameters())
                    )
                    _module._modules[name].adapter_up2.weight.requires_grad = True
                    _module._modules[name].adapter_down2.weight.requires_grad = True                    
                    names.append(name)
                elif _child_module.__class__.__name__ == "Conv2d" and name == "1.0" and "encoder" in father_name:
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = InjectedConv1x1(
                        _child_module.in_channels,
                        _child_module.out_channels,
                        _child_module.kernel_size,
                        _child_module.padding,
                        _child_module.dilation,
                        bias=_child_module.bias is not None,
                        r1=r1,
                        r2=r2,
                    )
                    _tmp.conv_adapter.weight = weight
                    if bias is not None:
                        _tmp.conv_adapter.bias = bias

                    idx1, idx2 = "1", "0"
                    # switch the module
                    _module._modules[idx1]._modules[idx2] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[idx1]._modules[idx2].adapter_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[idx1]._modules[idx2].adapter_down.parameters())
                    )
                    _module._modules[idx1]._modules[idx2].adapter_up.weight.requires_grad = True
                    _module._modules[idx1]._modules[idx2].adapter_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[idx1]._modules[idx2].adapter_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[idx1]._modules[idx2].adapter_down2.parameters())
                    )
                    _module._modules[idx1]._modules[idx2].adapter_up2.weight.requires_grad = True
                    _module._modules[idx1]._modules[idx2].adapter_down2.weight.requires_grad = True                    
                    names.append(name)
    
    return require_grad_params, names


if __name__ == '__main__':
    model = SAFNet()
    print(model)
    _, names = inject_trainable_adapter_cnn(model)
    print(model)
    print(names)
    x = torch.randn((1, 6, 512, 512))
    _, y = model(x, x, x)
    print(y.shape)

    # model.train()
    # x = torch.randn((1, 6, 512, 512))
    # _, y = model(x, x, x)
    # print(y.shape)
    # loss = (y - (y+0.1)).mean()
    # loss.backward()
    
    # h, w = 512, 512
    # H_patch, W_patch = 128, 128
    # x = torch.tensor([i for i in range(h*w*3*2)], dtype=torch.float32).reshape(2, 3, h, w)
    # print(x.shape)
    # y, B, C, H_blocks, W_blocks = window_paration(x, H_patch, W_patch)
    # print(y.shape)

    # output = window_inverse(y, B, C, H_blocks, W_blocks, H_patch, W_patch)
    # print(f"Restored shape: {output.shape}")  # 输出: torch.Size([2, 3, 512, 512])
    # assert torch.allclose(x, output), "Restoration failed!"

