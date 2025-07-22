import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import cv2
import copy

import config




def gen_anchors(feature_size, scale):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchors = np.array([0,0,15,15])


    # center of each cell
    xt = (base_anchors[0] + base_anchors[2]) / 2
    yt = (base_anchors[1] + base_anchors[3]) / 2

    # 10 base anchors, each has 4 coordinates x1, y1, x2, y2
    x1 = xt - widths / 2
    y1 = yt - heights / 2
    x2 = xt + widths / 2
    y2 = yt + heights / 2
    base_anchors = np.hstack((x1, y1, x2, y2))
    # print('base_anchors: ', base_anchors)
    
    # shift the base anchors to the feature map
    h, w = feature_size
    shift_x = np.arange(0, w)*scale
    shift_y = np.arange(0, h)*scale
    # print('shift_x: ', shift_x, 'shift_y: ', shift_y, 'scale: ', scale)
    # generate all anchors
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchors + [j, i, j, i])
    # print('(base_anchors + [j, i, j, i]: ', (base_anchors + [j, i, j, i]))
    # print(np.array(anchor).reshape((-1, 4)))
    return np.array(anchor).reshape((-1, 4))


def cal_overlap(base_anchors, gtboxes):
    """
    计算anchors和ground truth boxes之间的IOU
    
    Args:
        base_anchors: [N, 4] anchor boxes (x1, y1, x2, y2)
        gtboxes: [M, 4] ground truth boxes (x1, y1, x2, y2)
    
    Returns:
        overlaps: [N, M] IOU矩阵
    """
    gtboxes = np.array(gtboxes)
    
    # 计算所有anchor和gt box之间的IOU
    overlaps = np.zeros((base_anchors.shape[0], gtboxes.shape[0]))
    # print('overlaps.shape: ', overlaps.shape)
    
    for k in range(gtboxes.shape[0]):
        gt_box = gtboxes[k, :]  # 当前GT box [x1, y1, x2, y2]
        
        # 计算交集区域的坐标
        # 交集左上角：max(anchor_x1, gt_x1), max(anchor_y1, gt_y1)
        # 交集右下角：min(anchor_x2, gt_x2), min(anchor_y2, gt_y2)
        # print(base_anchors[:, 1], gt_box[1])
        inter_x1 = np.maximum(base_anchors[:, 0], gt_box[0])
        inter_y1 = np.maximum(base_anchors[:, 1], gt_box[1])
        inter_x2 = np.minimum(base_anchors[:, 2], gt_box[2])
        inter_y2 = np.minimum(base_anchors[:, 3], gt_box[3])
        
        # print('inter_x1: ', inter_x1, '\n', 'inter_x2: ', inter_x2, '\n', 'inter_y1: ', inter_y1, '\n', 'inter_y2: ', inter_y2)

        # 计算交集面积
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        # print('inter_w: ', inter_w, '\n', 'inter_h: ', inter_h)
        intersection = inter_w * inter_h

        # 计算anchor和gt box的面积
        anchor_areas = (base_anchors[:, 2] - base_anchors[:, 0]) * \
                      (base_anchors[:, 3] - base_anchors[:, 1])
        # print('anchor_areas: ', anchor_areas)
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        
        # 计算并集面积
        union = anchor_areas + gt_area - intersection
        # print('union: ', union, 'union.shape: ', union.shape)
        # 计算IOU，避免除零
        # print('intersection: ', intersection, 'intersection.shape: ', intersection.shape)
        overlaps[:, k] = intersection / (union + 1e-6)
        # print('overlaps: ', sorted(overlaps[:, k]))
    
    return overlaps


def bbox_transform(base_anchors, gtboxes):
    """
    计算从anchor box到ground truth box的回归目标
    
    Args:
        base_anchors: [N, 4] anchor boxes (x1, y1, x2, y2)
        gtboxes: [N, 4] ground truth boxes (x1, y1, x2, y2)
        N: number of anchor boxes
    
    Returns:
        bbox_targets: [N, 4] 回归目标 (dx, dy, dw, dh)
    """
    
    # 确保输入是numpy数组
    base_anchors = np.array(base_anchors, dtype=np.float32)
    gtboxes = np.array(gtboxes, dtype=np.float32)

    # print('base_anchors: ', base_anchors)
    # print('gtboxes: ', gtboxes)
    
    # 从边界坐标计算中心点坐标和宽高
    # Anchor boxes的中心点和宽高
    anchor_widths = base_anchors[:, 2] - base_anchors[:, 0] + 1.0
    anchor_heights = base_anchors[:, 3] - base_anchors[:, 1] + 1.0
    anchor_ctr_x = base_anchors[:, 0] + 0.5 * (anchor_widths - 1.0)
    anchor_ctr_y = base_anchors[:, 1] + 0.5 * (anchor_heights - 1.0)
    
    # Ground truth boxes的中心点和宽高
    gt_widths = gtboxes[:, 2] - gtboxes[:, 0] + 1.0
    gt_heights = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    gt_ctr_x = gtboxes[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gtboxes[:, 1] + 0.5 * (gt_heights - 1.0)
    
    # 计算回归目标
    # dx, dy: 中心点的相对偏移（归一化到anchor的宽高）
    dx = np.log(gt_widths / anchor_widths)
    dy = np.log(gt_heights / anchor_heights)
    
    # dw, dh: 宽高的对数缩放比例
    # dw = np.log(gt_widths / anchor_widths)
    # dh = np.log(gt_heights / anchor_heights)
    
    # 组合成回归目标 [dx, dy, dw, dh]
    bbox_targets = np.column_stack((dx, dy))
    
    return bbox_targets


def cal_rpn(img_size, feature_size, scale, gtboxes):
    h, w = img_size

    print('img_size: ', img_size, '\n', 'feature_size: ', feature_size, '\n', 'scale: ', scale)
    # gen base anchors
    base_anchors = gen_anchors(feature_size, scale)
    
    # print('base_anchors.shape: ', base_anchors.shape)
    # print('base_anchors.shape: ', base_anchors.shape)
    # print('gtboxes: ', gtboxes)
    # calculate iou between gtboxes and base_anchors
    overlaps = cal_overlap(base_anchors, gtboxes)
    # print('overlaps: ', overlaps)
    # init labels -1 don't care, 0 background, 1 foreground
    labels = np.empty(base_anchors.shape[0])
    labels.fill(-1)

    # for each gtbox,corresponds to an anchor which has highest iou
    # the ahchor with the highest iou overlap with a gtbox
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]


    # print('anchor_argmax_overlaps: ', anchor_argmax_overlaps)
    
    # print('gt_argmax_overlaps.shape: ', gt_argmax_overlaps.shape)
    # print('gt_argmax_overlaps: ', gt_argmax_overlaps)
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
    # 統計labels中各個取值的數量
    unique, counts = np.unique(labels, return_counts=True)
    # print("labels value counts:", dict(zip(unique, counts)))
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
    # print(anchor_max_overlaps)
    # print(gtboxes)
    # calculate bbox targets
    # print('anchor_argmax_overlaps.shape: ', anchor_argmax_overlaps.shape)
    # print('gtboxes.shape: ', gtboxes.shape)
    # print('gtboxes[anchor_argmax_overlaps, :].shape: ', gtboxes[anchor_argmax_overlaps, :].shape)
    # print('base_anchors.shape: ', base_anchors.shape)
    bbox_targets = bbox_transform(base_anchors, gtboxes[anchor_argmax_overlaps, :])
    # print('bbox_targets: ', bbox_targets)

    return [labels, bbox_targets], base_anchors



class ICDARDataset():
    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_list = os.listdir(img_dir)
        self.img_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        self.gt_list = os.listdir(gt_dir)
        self.gt_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.img_list)

        
    def __getitem__(self, index):
        """根据索引返回一个样本"""

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
        # 缩放图片至1600
        rescale_fac = max(h, w) / 1600
        if rescale_fac > 1:
            h = int(h/rescale_fac)
            w = int(w/rescale_fac)
            img = cv2.resize(img, (w, h))

        # 读取标注文件
        gt_path = os.path.join(self.gt_dir, 'gt_'+img_name)
        gtbox = self.parse_gtfile(gt_path, rescale_fac)

        # 水平翻转
        rd = np.random.random()
        if rd < 0.3:
            # 水平翻转, 參數1
            img = cv2.flip(img, 1)
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        # 垂直翻转
        if rd > 0.3 and rd < 0.6:
            # 垂直翻转, 參數0
            img = cv2.flip(img, 0)
            newy1 = h - gtbox[:, 3] - 1
            newy2 = h - gtbox[:, 1] - 1
            gtbox[:, 1] = newy1
            gtbox[:, 3] = newy2
        
        # # 旋轉
        # if rd > 0.4 and rd < 0.6:
        #     angle = np.random.randint(-10, 10)
        #     # 旋轉矩陣
        #     M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        #     # 旋轉圖片
        #     img = cv2.warpAffine(img, M, (w, h))
        #     # 旋轉gtbox
        #     gtbox = cv2.transform(gtbox.reshape(1, -1, 2), M).reshape(-1, 2)


        # 生成RPN标签
        [cls, regr], base_anchors = cal_rpn(
            (h, w),
            (int(h/16), int(w/16)),
            16,
            gtbox
            )

        m_img = img - [123.68, 116.78, 103.94]

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        # transform to tensor
        # img = torch.from_numpy(img.transpose([2, 0, 1])).float()
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        regr = torch.from_numpy(regr).float()
        cls = torch.from_numpy(cls).float()

        return m_img, cls, regr

    
    def box_transfer_v2(self, coor_lists, rescale_fac = 1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16*i-0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)
    
    # 解析标注文件
    def parse_gtfile(self, gt_path, rescale_fac):
            if '.jpg' in gt_path:
                gt_path = gt_path.replace('.jpg', '.txt')
            if '.png' in gt_path:
                gt_path = gt_path.replace('.png', '.txt')
            if '.gif' in gt_path:
                gt_path = gt_path.replace('.gif', '.txt')

            print('rescale_fac: ', rescale_fac)
            print('gt_path: ', gt_path)

            with open(gt_path, 'r', encoding="utf-8-sig") as f:
                gt = f.readlines()
            gtbox = []
            for line in gt:
                # 切割成
                line = line.strip().split(',')
                # print(line)
                x1, y1, x2, y2, x3, y3, x4, y4 = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7])

                gtbox.append([x1, y1, x2, y2, x3, y3, x4, y4])
                # print([_x1, _y1, _x2, _y2])
                
            gtbox = np.array(gtbox)
            gtbox = self.box_transfer_v2(gtbox, rescale_fac)

            return gtbox


    
    








