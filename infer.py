# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:08:12 2022

@author: shyoh
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import canvas
from model import generator
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

torch.manual_seed(0)
gpu = torch.cuda.is_available()
device_ids = [0, 1, 2, 3]
device = torch.device(f"cuda:{device_ids[0]}" if gpu else "cpu")

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
    
def random_init(batch, max_elem):
    coef = [0.1, 0.8, 1, 1]
    cls_1 = torch.tensor(np.random.choice(4, size=(batch, max_elem, 1), p=np.array(coef) / sum(coef)))
    cls = torch.zeros((batch, max_elem, 4))
    cls.scatter_(-1, cls_1, 1)
    box_xyxy = torch.normal(0.5, 0.15, size=(batch, max_elem, 1, 4))
    box = box_xyxy_to_cxcywh(box_xyxy)
    init_layout = torch.concat([cls.unsqueeze(2), box], dim=2)
    return init_layout

def test(G, testing_dl, epoch_n):
    global fix_noise, no
    G.eval()
    clses = []
    boxes = []
    with torch.no_grad():
        for imgs in testing_dl:
            imgs = imgs.to(device)
            cls, box = G(imgs, fix_noise)
            clses.append(torch.argmax(cls.detach().cpu(), dim=-1, keepdim=True))
            boxes.append(box_cxcywh_to_xyxy(box.detach().cpu()))            
    clses = torch.concat(clses, dim=0).numpy()
    boxes = torch.concat(boxes, dim=0).numpy()
    
    torch.save(clses, f"output/clses-Epoch300.pt")
    torch.save(boxes, f"output/boxes-Epoch300.pt")
    
def main():
    global fix_noise, no
    test_bg_path = "Dataset/test/image_canvas"
    test_sal_dir_1 = "Dataset/test/saliencymaps_pfpn"
    test_sal_dir_2 = "Dataset/test/saliencymaps_basnet"
    test_batch_size = 4
    
    ckpt_path = "output/DS-GAN-Epoch300.pth"
    
    testing_set = canvas(test_bg_path, test_sal_dir_1, test_sal_dir_2, train=False)
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)
    
    max_elem = 32
    args_g = {
        "backbone": "resnet50",
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": max_elem * 8,
        "num_layers": 4,
        "output_size": 8,
        "max_elem": max_elem
    }
    
    print(f"testing_set: {len(testing_set)}")
    fix_noise = random_init(test_batch_size, max_elem)
    
    G = generator(args_g)
    ckpt = torch.load(ckpt_path)
        
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    
    G.load_state_dict(new_state_dict)
    
    if gpu:
        G = G.to(device)
        G = torch.nn.DataParallel(G, device_ids=device_ids)
        
    test(G, testing_dl, 1)

if __name__ == "__main__":
    main()
    