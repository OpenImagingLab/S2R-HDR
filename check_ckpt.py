import torch
import os

root_dir = "pretrained_models/original"
save_dir = "pretrained_models"
for ckpt in os.listdir(root_dir):
    checkpoint = torch.load(os.path.join(root_dir, ckpt))
    print(checkpoint.keys(), ckpt)
    save_dict = {}
    for k, v in checkpoint.items():
        if k == "epoch":
            continue
        save_dict[k] = v
    torch.save(save_dict, os.path.join(save_dir, ckpt))
