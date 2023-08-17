import numpy as np
import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.array(inter) / np.array(union)
    return iou

def reorder(cls, box, o="xyxy", max_elem=None):
    if o == "cxcywh":
        box = box_cxcywh_to_xyxy(box)
    if max_elem == None:
        max_elem = len(cls)
    # init
    order = []
    
    # convert
    cls = np.array(cls)
    area = box_area(box)
    order_area = sorted(list(enumerate(area)), key=lambda x: x[1], reverse=True)
    iou = box_iou(box, box)
    
    # arrange
    text = np.where(cls == 1)[0]
    logo = np.where(cls == 2)[0]
    deco = np.where(cls == 3)[0]
    order_text = sorted(np.array(list(enumerate(area)))[text].tolist(), key=lambda x: x[1], reverse=True)
    order_deco = sorted(np.array(list(enumerate(area)))[deco].tolist(), key=lambda x: x[1])
    
    # deco connection
    connection = {}
    reverse_connection = {}
    for idx, _ in order_deco:
        con = []
        for idx_ in logo:
            if iou[idx, idx_]:
                connection[idx_] = idx
                con.append(idx_)
        for idx_ in text:
            if iou[idx, idx_]:
                connection[idx_] = idx
                con.append(idx_)
        for idx_ in deco:
            if idx == idx_: continue
            if iou[idx, idx_]:
                if idx_ not in connection:
                    connection[idx_] = [idx]
                else:
                    connection[idx_].append(idx)
                con.append(idx_)
        reverse_connection[idx] = con
                    
    # reorder
    for idx in logo:
        if idx in connection:
            d = connection[idx]
            d_group = reverse_connection[d]
            for idx_ in d_group:
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            order.append(idx)
    for idx, _ in order_text:
        if len(order) >= max_elem:
            break
        if idx in connection:
            d = connection[idx]
            d_group = reverse_connection[d]
            for idx_ in d_group:
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            order.append(idx)
            
    if len(order) < max_elem:
        non_obj = np.where(cls == 0)[0]
        order.extend(non_obj)
    
    return order[:min(len(cls), max_elem)]