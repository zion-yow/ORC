import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import cv2

import config


def parse_gtfile(gt_path, rescale_fac):
        with open(gt_path, 'r') as f:
            gt = f.readlines()
        gtbox = []
        for line in gt:
            line = line.strip().split(' ')
            x1, y1, x2, y2 = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            x1 = x1 / rescale_fac
            y1 = y1 / rescale_fac
            x2 = x2 / rescale_fac
            y2 = y2 / rescale_fac
            gtbox.append([x1, y1, x2, y2])
        gtbox = np.array(gtbox)
        
        return gtbox


def gen_anchors(feature_size, scale):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]*10

    # gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchors = np.array([0,0,15,15])
    # center of each cell
    xt = (base_anchors[:, 0] + base_anchors[:, 2]) / 2
    yt = (base_anchors[:, 1] + base_anchors[:, 3]) / 2

    # 10 base anchors, each has 4 coordinates x1, y1, x2, y2
    x1 = xt - widths / 2
    y1 = yt - heights / 2
    x2 = xt + widths / 2
    y2 = yt + heights / 2
    base_anchors = np.hstack((x1, y1, x2, y2))
    
    # shift the base anchors to the feature map
    h, w = feature_size
    shift_x = np.arange(0, w*scale)
    shift_y = np.arange(0, h*scale)

    # generate all anchors
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchors + [j, i, j, i])

    
    return np.array(anchor).reshape((-1, 4))

def cal_overlap(base_anchors, gtboxes):
    gtboxes = np.array(gtboxes)

    # calculate iou between base_anchors and gtboxes
    overlaps = np.zeros((base_anchors.shape[0], gtboxes.shape[0]))
    for k in range(gtboxes.shape[0]):
        gt = gtboxes[k, :]
        gt = np.tile(gt, (base_anchors.shape[0], 1))
        gt_widths = gt[:, 2] - gt[:, 0] + 1
        gt_heights = gt[:, 3] - gt[:, 1] + 1
        gt_ctr_x = gt[:, 0] + 0.5 * (gt_widths - 1)
        gt_ctr_y = gt[:, 1] + 0.5 * (gt_heights - 1)

        # calculate iou between base_anchors and gt
        base_anchors_widths = base_anchors[:, 2] - base_anchors[:, 0] + 1
        base_anchors_heights = base_anchors[:, 3] - base_anchors[:, 1] + 1
        base_anchors_ctr_x = base_anchors[:, 0] + 0.5 * (base_anchors_widths - 1)
        base_anchors_ctr_y = base_anchors[:, 1] + 0.5 * (base_anchors_heights - 1)

        base_anchors_widths = base_anchors_widths.reshape(-1, 1)
        base_anchors_heights = base_anchors_heights.reshape(-1, 1)
        base_anchors_ctr_x = base_anchors_ctr_x.reshape(-1, 1)
        base_anchors_ctr_y = base_anchors_ctr_y.reshape(-1, 1)
        

def bbox_transform(base_anchors, gtboxes):
    # bbox is the distance between anchor and gtbox at the edge



def cal_rpn(img_size, feature_size, stride, gtboxes):
    h, w = img_size

    # gen base anchors
    base_anchors = gen_anchors(feature_size, scale)

    # calculate iou between gtboxes and base_anchors
    overlaps = cal_overlap(base_anchors, gtboxes)

    # init labels -1 don't care, 0 background, 1 foreground
    labels = np.empty(base_anchors.shape[0])
    labels.fill(-1)

    # for each gtbox,corresponds to an anchor which has highest iou
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # the ahchor with the highest iou overlap with a gtbox
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # iou > IOU_POSITIVE, label = 1
    labels[anchor_max_overlaps >= config.IOU_POSITIVE] = 1

    # iou < IOU_NEGATIVE, label = 0
    labels[anchor_max_overlaps < config.IOU_NEGATIVE] = 0

    # ensure that each gtbox has at least one foreground anchor
    labels[gt_argmax_overlaps] = 1

    # check anchors outside image
    outside_anchor = np.where(
        (base_anchors[:, 0] < 0) |  
        (base_anchors[:, 1] < 0) |
        (base_anchors[:, 2] > w) |
        (base_anchors[:, 3] > h)
    )[0]

    labels[outside_anchor] = -1

    # subsample foreground labels, if greater than RPN_FORE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if len(fg_index) > config.RPN_FORE_NUM:
        labels[np.random.choice(fg_index, len(fg_index) - config.RPN_FORE_NUM, replace=False)] = -1
    
    # subsample background labels
    # if not config.OHEM:
    #     bg_index = np.where(labels == 0)[0]
    #     num_bg = 
    #     if len(bg_index) > config.RPN_BACK_NUM:
    #         labels[np.random.choice(bg_index, len(bg_index) - config.RPN_BACK_NUM, replace=False)] = -1
    
    # calculate bbox targets
    bbox_targets = bbox_transform(base_anchors, gtboxes[gt_argmax_overlaps, :])


    return [labels, bbox_targets], base_anchors

class ICDARDataset():
    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_list = os.listdir(img_dir)
        self.gt_list = os.listdir(gt_dir)

        
    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(img_path)

        if img is None:
            print(img_path)
            with open('error_img.txt', 'a') as f:
                f.write('{}\n'.format(img_path))
            img_name = 'img_4929.jpg'
            img_path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_path)

        h, w, c = img.shape
        rescale_fac = max(h, w) / 1600
        if rescale_fac > 1:
            h = int(h/rescale_fac)
            w = int(w/rescale_fac)
            img = cv2.resize(img, (w, h))

        gt_path = os.path.join(self.gt_dir, img_name.replace('.jpg', '.txt'))
        gtbox = self.parse_gtfile(gt_path, rescale_fac)

        # random flip
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2
         
        [cls, regr], base_anchors = cal_rpn(
            (h, w),
            (int(h/16), int(w/16)),
            16,
            gtbox
            )
        
        m_img = img - IMAGE_MEAN
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        # transform to tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        regr = torch.from_numpy(regr).float()
        cls = torch.from_numpy(cls).float()

        return m_img, cls, regr
    

    
    








