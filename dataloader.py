#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:13:15 2022

@author: kinsleyhsu
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pandas import read_csv
from PIL import Image
from designSeq import reorder

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class canvasLayout(Dataset):
    def __init__(self, inp_dir, sal_dir_1, sal_dir_2, csv_path, max_elem=8):
        img = os.listdir(inp_dir)
        self.inp = list(map(lambda x: os.path.join(inp_dir, x), img))
        self.sal_1 = list(map(lambda x: os.path.join(sal_dir_1, x.replace(".png", "_pred.png")), img))
        self.sal_2 = list(map(lambda x: os.path.join(sal_dir_2, x), img))
        
        df = read_csv(csv_path)
        self.max_elem = max_elem
        self.poster_name = list(map(lambda x: "train/" + x.replace("_mask", ""), img))
        self.groups = df.groupby(df.poster_path)
        
        self.transform = transforms.Compose([
            transforms.Resize([350, 240]),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.inp)
     
    def __getitem__(self, idx):
        img_inp = Image.open(self.inp[idx]).convert("RGB")
        img_sal_1 = Image.open(self.sal_1[idx]).convert("L")
        img_sal_2 = Image.open(self.sal_2[idx]).convert("L")
        img_sal = Image.fromarray(np.maximum(np.array(img_sal_1), np.array(img_sal_2)))
         
        img_inp = self.transform(img_inp)
        img_sal = self.transform(img_sal)
         
        cc = torch.concat([img_inp, img_sal])
         
        label = np.zeros((self.max_elem, 2, 4))
        sliced_df = self.groups.get_group(self.poster_name[idx])
        cls = list(sliced_df["cls_elem"])
        box = torch.tensor(list(map(eval, sliced_df["box_elem"])))
        
        order = reorder(cls, box, "xyxy", self.max_elem)
        
        for i in range(len(order)):
            idx = order[i]
            label[i][0][int(cls[idx])] = 1
            label[i][1] = box[idx]
            if label[i][1][0] > label[i][1][2] or label[i][1][1] > label[i][1][3]:
                label[i][1][:2], label[i][1][2:] = label[i][1][2:], label[i][1][:2]
            label[i][1] = box_xyxy_to_cxcywh(torch.tensor(label[i][1]))
            label[i][1][::2] /= 513
            label[i][1][1::2] /= 750
        for i in range(len(order), self.max_elem):
            label[i][0][0] = 1
        
        return cc, torch.tensor(label).float()
    
class canvas(Dataset):
    def __init__(self, bg_dir, sal_dir_1, sal_dir_2, train=True):
        img = os.listdir(bg_dir)
        if train:
            img = img[:4]
        else:
            torch.save(img, "test_order.pt")
        self.bg = list(map(lambda x: os.path.join(bg_dir, x), img))
        self.sal_1 = list(map(lambda x: os.path.join(sal_dir_1, x.replace(".png", "_pred.png")), img))
        self.sal_2 = list(map(lambda x: os.path.join(sal_dir_2, x), img))
        
        self.transform = transforms.Compose([
            transforms.Resize([350, 240]),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.bg)
    
    def __getitem__(self, idx):
        img_bg = Image.open(self.bg[idx]).convert("RGB")
        img_sal_1 = Image.open(self.sal_1[idx]).convert("L")
        img_sal_2 = Image.open(self.sal_2[idx]).convert("L")
        img_sal = Image.fromarray(np.maximum(np.array(img_sal_1), np.array(img_sal_2)))
        
        img_bg = self.transform(img_bg)
        img_sal = self.transform(img_sal)
        
        cc = torch.concat([img_bg, img_sal])
        return cc
    
def main():
    test_batch_size = 4
    test_bg_dir = "../Dataset/640_canvas"
    test_sal_dir_1 = "../Dataset/640_sal"
    test_sal_dir_2 = "../Dataset/640_sal_2"
    train_batch_size = 32
    train_inp_dir = "../Dataset/inp"
    train_sal_dir_1 = "../Dataset/sal"
    train_sal_dir_2 = "../Dataset/sal_2"
    train_csv_path = "../Dataset/train_csv_9973.csv"
    mex_elem = 8
    
    testing_set = canvas(test_bg_dir, test_sal_dir_1, test_sal_dir_2)
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)
    
    img = next(iter(testing_dl))
    print(img.shape)
    
    training_set = canvasLayout(train_inp_dir, train_sal_dir_1, train_sal_dir_2, train_csv_path, mex_elem)
    training_dl = DataLoader(training_set, num_workers=16, batch_size=train_batch_size, shuffle=True)
    img, label = next(iter(training_dl))
    print(img.shape, label.shape)
    print(label[0])

if __name__ == "__main__":
    import time
    cur = time.time()
    main()
    end = time.time()
    print(end-cur)