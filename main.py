#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 02:33:37 2022

@author: kinsleyhsu
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import canvasLayout, canvas
import numpy as np
from model import generator, discriminator
from RecLoss import SetCriterion, HungarianMatcher
from utils import setup_seed
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

setup_seed(0)
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
    global coef
    cls_1 = torch.tensor(np.random.choice(4, size=(batch, max_elem, 1), p=np.array(coef) / sum(coef)))
    cls = torch.zeros((batch, max_elem, 4))
    cls.scatter_(-1, cls_1, 1)
    box_xyxy = torch.normal(0.5, 0.15, size=(batch, max_elem, 1, 4))
    box = box_xyxy_to_cxcywh(box_xyxy)
    init_layout = torch.concat([cls.unsqueeze(2), box], dim=2)
    return init_layout
    

def train(G, D, training_dl, criterionRec, criterionAdv, w_criterionAdv, optimizerG, optimizerD, schedulerG, schedulerD, epoch_n, max_elem):
    for idx, (image, label) in enumerate(training_dl):
        b_s = image.size(0)
        image, label = image.to(device), label.to(device)
        all_real = torch.ones(image.size(0), dtype=torch.float).to(device)
        all_fake = torch.full((image.size(0),), -1, dtype=torch.float).to(device)
        init_layout = random_init(b_s, max_elem).to(device)
        
        G.train()
        D.train()
        
        # update G
        G.zero_grad()
        cls, box = G(image, init_layout)
        label_f = torch.concat([cls.unsqueeze(2), box.unsqueeze(2)], dim=2)
        outputG = D(image, label_f)
        D_G_z1 = outputG.mean()
        
        cls_gt = label[:, :, 0]
        box_gt = label[:, :, 1]
        outputs = {
            "pred_logits": cls,
            "pred_boxes": box.float()
        }
        targets = [{
            "labels": c.long(), 
            "boxes": b.float()
        } for c, b in zip(torch.argmax(cls_gt, dim=-1), box_gt)]
        
        lossG = criterionAdv(outputG.view(-1), all_real)
        lossRec = sum(criterionRec(outputs, targets).values())
        lossesG = w_criterionAdv * lossG + lossRec
        lossesG.backward()
        optimizerG.step()
        
        # update D
        D.zero_grad()
        outputD_f = D(image, label_f.detach())
        lossD_f = criterionAdv(outputD_f.view(-1), all_fake)
        lossRec = sum(criterionRec(outputs, targets).values())
        D_G_z2 = outputD_f.mean()
        
        outputD_r = D(image, label)
        lossD_r = criterionAdv(outputD_r.view(-1), all_real)
        D_x = outputD_r.mean()
        
        lossesD = w_criterionAdv * (lossD_r + lossD_f)
        lossesD.backward()
        optimizerD.step()
                
        print('[%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch_n, idx, lossesD, lossesG, D_x, D_G_z1, D_G_z2))
    
    schedulerD.step()
    schedulerG.step()
        

def test(G, testing_dl, epoch_n, img_path):
    global fix_init_layout
    G.eval()
    
    if epoch_n == 300: torch.save(G.state_dict(), f"output/DS-GAN-Epoch300.pth")
    for imgs in testing_dl:
        imgs = imgs.to(device)
        cls, box = G(imgs, fix_init_layout)
        plt.figure(figsize=(12, 5))
        for idx, (c, b) in enumerate(zip(cls, box)):
            c = c.detach().cpu()
            c = torch.argmax(c, dim=1)
            b = box_cxcywh_to_xyxy(b.detach().cpu())
            print(c, b)
            b[:, ::2] *= 513
            b[:, 1::2] *= 750
            img = Image.open(img_path[idx]).convert("RGB")
            drawn = draw_box_f(img, c.numpy(), b.numpy())
            plt.subplot(1, 4, idx+1)
            plt.axis("off")
            plt.imshow(drawn)
        plt.savefig(f"output/training_plot/Epoch{epoch_n}")            
        return

def draw_box_f(img, cls_list, box_list):
    img_copy = img.copy()
    draw = ImageDraw.ImageDraw(img_copy)
    cls_color_dict = {0: "black", 1: 'green', 2: 'red', 3: 'orange'}
    for cls, box in zip(cls_list, box_list):
        draw.rectangle(box, fill=None, outline=cls_color_dict[int(cls)], width=5)
    return img_copy
  
def isbase(name):
    if name.startswith("module.resnet_fpn"):
        return True
    return False
    
def main():
    global fix_init_layout, coef
    train_inp_dir = "Dataset/train/inpainted_poster"
    train_sal_dir_1 = "Dataset/train/saliencymaps_pfpn"
    train_sal_dir_2 = "Dataset/train/saliencymaps_basnet"
    train_csv_path = "Dataset/train_csv_9973.csv"
    test_bg_dir = "Dataset/test/image_canvas"
    test_sal_dir_1 = "Dataset/test/saliencymaps_pfpn"
    test_sal_dir_2 = "Dataset/test/saliencymaps_basnet"
    
    train_batch_size = 128
    test_batch_size = 4
    max_elem = 32
    
    epoch = 300
    linear_step = 100
    
    training_set = canvasLayout(train_inp_dir, train_sal_dir_1, train_sal_dir_2, train_csv_path, max_elem)
    training_dl = DataLoader(training_set, num_workers=16, batch_size=train_batch_size, shuffle=True)
    
    testing_set = canvas(test_bg_dir, test_sal_dir_1, test_sal_dir_2)
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)
    
    args_g = {
        "backbone": "resnet50",
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": max_elem * 8,
        "num_layers": 4,
        "output_size": 8,
        "max_elem": max_elem
    }
    args_d = {
        "backbone": "resnet18",
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": max_elem * 8,
        "num_layers": 2,
        "output_size": 8,
        "max_elem": max_elem
    }
    
    print(f"training_set: {len(training_set)}, testing_set: {len(testing_set)}")
    
    G = generator(args_g)
    D = discriminator(args_d)
    if gpu:
        G = G.to(device)
        G = torch.nn.DataParallel(G, device_ids=device_ids)
        D = D.to(device)
        D = torch.nn.DataParallel(D, device_ids=device_ids)
    
    criterionAdv = nn.HingeEmbeddingLoss()
    matcher = HungarianMatcher(2, 5, 2)
    weight_dict = {
        "loss_ce": 2, "loss_bbox": 5, "loss_giou": 2
    }
    coef = [0.1, 0.8, 1, 1]
    criterionRec = SetCriterion(3, matcher, weight_dict, coef, ['labels', 'boxes']).to(device)
    
    fix_init_layout = random_init(test_batch_size, max_elem)
    
    paramsG = list(filter(lambda kv: not isbase(kv[0]), G.named_parameters()))
    paramsD = list(filter(lambda kv: not isbase(kv[0]), D.named_parameters()))
    base_paramsG = list(filter(lambda kv: isbase(kv[0]), G.named_parameters()))
    base_paramsD = list(filter(lambda kv: isbase(kv[0]), D.named_parameters()))
    paramsG = list(map(lambda kv: kv[1], paramsG))
    paramsD = list(map(lambda kv: kv[1], paramsD))
    base_paramsG = list(map(lambda kv: kv[1], base_paramsG))
    base_paramsD = list(map(lambda kv: kv[1], base_paramsD))
    
    optimizerG = optim.Adam([{"params": paramsG, "lr": 1e-4},
                            {"params": base_paramsG, "lr": 1e-5}
                            ])
    optimizerD = optim.Adam([{"params": paramsD, "lr": 1e-3},
                            {"params": base_paramsD, "lr": 1e-4}
                            ])
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=torch.arange(0, epoch, 50), gamma=0.8)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=torch.arange(0, epoch, 25), gamma=0.8)
    
    try:
        os.makedirs("output/training_plot")
    except:
        pass
    
    for e in range(1, epoch + 1):
        if e > linear_step:
            w_L_adv = 1
        else:
            w_L_adv = 1 / linear_step * (e - 1)
        train(G, D, training_dl, criterionRec, criterionAdv, w_L_adv, optimizerG, optimizerD, schedulerG, schedulerD, e, max_elem)
        test(G, testing_dl, e, testing_set.bg)
        

if __name__ == "__main__":
    main()