# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:18:00 2022

@author: shyoh
"""

import torch
import os
import copy
import numpy as np
import cv2
from PIL import Image, ImageDraw
from math import log
from collections import OrderedDict

gpu = torch.cuda.is_available()
device_ids = [0, 1, 2, 3]
device = torch.device(f"cuda:{device_ids[0]}" if gpu else "cpu")

def draw_box(img, elems, elems2):
    drawn_outline = img.copy()
    drawn_fill = img.copy()
    draw_ol = ImageDraw.ImageDraw(drawn_outline)
    draw_f = ImageDraw.ImageDraw(drawn_fill)
    cls_color_dict = {1: 'green', 2: 'red', 3: 'orange'}
    
    for cls, box in elems:
        if cls[0]:
            draw_ol.rectangle(tuple(box), fill=None, outline=cls_color_dict[cls[0]], width=5)
    
    s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
    for cls, box in s_elems:
        if cls[0]:
            draw_f.rectangle(tuple(box), fill=cls_color_dict[cls[0]])
            
    drawn_outline = drawn_outline.convert("RGBA")
    drawn_fill = drawn_fill.convert("RGBA")
    drawn_fill.putalpha(int(256 * 0.3))
    drawn = Image.alpha_composite(drawn_outline, drawn_fill)
    
    return drawn

def cvt_pilcv(img, req='pil2cv', color_code=None):
    if req == 'pil2cv':
        if color_code == None:
            color_code = cv2.COLOR_RGB2BGR
        dst = cv2.cvtColor(np.asarray(img), color_code)
    elif req == 'cv2pil':
        if color_code == None:
            color_code = cv2.COLOR_BGR2RGB
        dst = Image.fromarray(cv2.cvtColor(img, color_code))
    return dst

def img_to_g_xy(img):
    img_cv_gs = np.uint8(cvt_pilcv(img, "pil2cv", cv2.COLOR_RGB2GRAY))
    # Sobel(src, ddepth, dx, dy)
    grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
    grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
    grad_xy = ((grad_x ** 2 + grad_y ** 2) / 2) ** 0.5
    grad_xy = grad_xy / np.max(grad_xy) * 255
    img_g_xy = Image.fromarray(grad_xy).convert('L')
    return img_g_xy

def metrics_iou(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    
    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2
    
    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter =  min(yr_1, yr_2) - max(yl_1, yl_2)
 
    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0
 
    return a_inter / (a_1 + a_2 - a_inter)

def metrics_inter_oneside(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    
    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2
    
    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter =  min(yr_1, yr_2) - max(yl_1, yl_2)
 
    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0
 
    return a_inter / a_2

def metrics_val(img_size, clses, boxes):
    """
    The ratio of non-empty layouts.
    Higher is better.
    """
    w, h = img_size
    
    total_elem = 0
    empty_elem = 0
    
    for cls, box in zip(clses, boxes):
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        total_elem += len(mask_box)
        for mb in mask_box:
            xl, yl, xr, yr = mb
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(513, xr)
            yr = min(750, yr)
            if abs((xr - xl) * (yr - yl)) < 5.13 * 7.50 * 10:
                empty_elem += 1
    return 1 - empty_elem / total_elem

def getRidOfInvalid(img_size, clses, boxes):
    w, h = img_size
    
    for i, (cls, box) in enumerate(zip(clses, boxes)):
        for j, b in enumerate(box):
            xl, yl, xr, yr = b
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(513, xr)
            yr = min(750, yr)
            if abs((xr - xl) * (yr - yl)) < 5.13 * 7.50 * 10:
                if clses[i, j]:
                    clses[i, j] = 0
    return clses

def metrics_uti(img_names, clses, boxes):
    metrics = 0
    for idx, name in enumerate(img_names):
        pic_1 = np.array(Image.open(os.path.join("Dataset/test/saliencymaps_pfpn", 
                                               name.replace(".", "_pred."))).convert("L").resize((513, 750))) / 255
        pic_2 = np.array(Image.open(os.path.join("Dataset/test/saliencymaps_basnet", 
                                               name)).convert("L").resize((513, 750))) / 255
        pic = np.maximum(pic_1, pic_2)
        c_pic = np.ones_like(pic) - pic
        
        cal_mask = np.zeros_like(pic)
        
        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)
        
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        
        total_not_sal = np.sum(c_pic)
        total_utils = np.sum(c_pic * cal_mask)
        
        if total_not_sal and total_utils:
            metrics += (total_utils / total_not_sal)
    return metrics / len(img_names)

def metrics_rea(img_names, clses, boxes):
    '''
    Average gradients of the pixels covered by predicted text-only elements.
    Lower is better.
    '''
    metrics = 0
    for idx, name in enumerate(img_names):
        pic = Image.open(os.path.join("Dataset/test/image_canvas", name)).convert("RGB").resize((513, 750))
        img_g_xy = np.array(img_to_g_xy(pic)) / 255
        cal_mask = np.zeros_like(img_g_xy)
        
        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)
        
        text = (cls == 1).reshape(-1)
        text_box = box[text]
        deco = (cls == 3).reshape(-1)
        deco_box = box[deco]
        
        for mb in text_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        for mb in deco_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 0
        
        total_area = np.sum(cal_mask)
        total_grad = np.sum(img_g_xy[cal_mask == 1])
        if total_grad and total_area:
            metrics += (total_grad / total_area)
    return metrics / len(img_names)

def metrics_ove(clses, boxes):
    """
    Ratio of overlapping area.
    Lower is better.
    """
    metrics = 0
    for cls, box in zip(clses, boxes):
        ove = 0
        mask = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        mask_box = box[mask]
        n = len(mask_box)
        for i in range(n):
            bb1 = mask_box[i]
            for j in range(i + 1, n):
                bb2 = mask_box[j]
                ove += metrics_iou(bb1, bb2)
        metrics += ove / n
    return metrics / len(clses)

def metrics_und_l(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        mask_deco = (cls == 3).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        n1 = len(box_deco)
        n2 = len(box_other)
        if n1:
            avali += 1
            for i in range(n1):
                max_ios = 0
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    ios = metrics_inter_oneside(bb1, bb2)
                    max_ios = max(max_ios, ios)
                und += max_ios
            metrics += und / n1
    if avali > 0:
        return metrics / avali
    return 0

def is_contain(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    
    c1 = xl_1 <= xl_2
    c2 = yl_1 <= yl_2
    c3 = xr_2 >= xr_2
    c4 = yr_1 >= yr_2
 
    return c1 and c2 and c3 and c4

def metrics_und_s(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        mask_deco = (cls == 3).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        n1 = len(box_deco)
        n2 = len(box_other)
        if n1:
            avali += 1
            for i in range(n1):
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    if is_contain(bb1, bb2):
                        und += 1
                        break
            metrics += und / n1
    if avali > 0:
        return metrics / avali
    return 0

def ali_g(x):
    return -log(1 - x, 10)

def ali_delta(xs):
    n = len(xs)
    min_delta = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(xs[i] - xs[j])
            min_delta = min(min_delta, delta)
    return min_delta

def metrics_ali(clses, boxes):
    """
    Indicator of the extent of non-alignment of pairs of elements.
    Lower is better.
    """
    metrics = 0
    for cls, box in zip(clses, boxes):
        ali = 0
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        
        theda = []
        for mb in mask_box:
            pos = copy.deepcopy(mb)
            pos[0] /= 513
            pos[2] /= 513
            pos[1] /= 750
            pos[3] /= 750
            theda.append([pos[0], pos[1], (pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2, pos[2], pos[3]])
        theda = np.array(theda)
        if theda.shape[0] <= 1:
            continue
        
        n = len(mask_box)
        for i in range(n):
            g_val = []
            for j in range(6):
                xys = theda[:, j]
                delta = ali_delta(xys)
                g_val.append(ali_g(delta))
            ali += min(g_val)
        metrics += ali

    return metrics / len(clses)

def metrics_occ(img_names, clses, boxes):
    '''
    Average saliency of the pixels covered.
    Lower is better.
    '''
    metrics = 0
    for idx, name in enumerate(img_names):
        pic_1 = np.array(Image.open(os.path.join("Dataset/test/saliencymaps_pfpn", 
                                               name.replace(".", "_pred."))).convert("L").resize((513, 750))) / 255
        pic_2 = np.array(Image.open(os.path.join("Dataset/test/saliencymaps_basnet", 
                                               name)).convert("L").resize((513, 750))) / 255
        pic = np.maximum(pic_1, pic_2)
        cal_mask = np.zeros_like(pic)
        
        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)
        
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        
        total_area = np.sum(cal_mask)
        total_sal = np.sum(pic[cal_mask == 1])
        if total_sal and total_area:
            metrics += (total_sal / total_area)
    return metrics / len(img_names)
        
def save_figs(names, clses, boxes, save_dir):
    try:
        os.makedirs(save_dir)
    except:
        pass
    for idx, name in enumerate(names):
        pic = Image.open(os.path.join("Dataset/test/image_canvas", name)).convert("RGB").resize((513, 750))
        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)
        drawn = draw_box(pic, zip(cls, box), zip(cls, box))
        drawn.save(os.path.join(save_dir, name.replace(".", "_.")))

def main():
    no = 1
    save_dir = f"output/result_plot/"
    names = torch.load("test_order.pt")
    print("len:", len(names))
    clses = torch.load("output/clses-Epoch300.pt")
    boxes = torch.load("output/boxes-Epoch300.pt")
    boxes[:, :, ::2] *= 513
    boxes[:, :, 1::2] *= 750
    
    #save_figs(names, clses, boxes, save_dir)
    
    print("metrics_val:", metrics_val((513, 750), clses, boxes))
    clses = getRidOfInvalid((513, 750), clses, boxes)
    
    print("metrics_ove:", metrics_ove(clses, boxes))
    print("metrics_ali:", metrics_ali(clses, boxes))
    print("metrics_und_l:", metrics_und_l(clses, boxes))
    print("metrics_und_s:", metrics_und_s(clses, boxes))
    
    print("metrics_uti:", metrics_uti(names, clses, boxes))
    print("metrics_occ:", metrics_occ(names, clses, boxes))
    print("metrics_rea:", metrics_rea(names, clses, boxes))
     
if __name__ == "__main__":
    main()