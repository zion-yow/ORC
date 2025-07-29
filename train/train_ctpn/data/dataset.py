import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import cv2
import copy
import time

import config




def gen_anchors(feature_size, scale):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    h, w = feature_size

    # feature map每個像素點的x,y坐標
    shift_x = np.arange(0, w)*scale
    shift_y = np.arange(0, h)*scale
    # 基礎anchor
    base_anchors = np.array([0,0,15,15])
    # 基礎anchor的中心點坐標
    xt = (base_anchors[0] + base_anchors[2]) / 2
    yt = (base_anchors[1] + base_anchors[3]) / 2
    # 生成k=9個anchor, 每個anchor有4個坐標 x1, y1, x2, y2
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)
    x1 = xt - widths / 2
    y1 = yt - heights / 2
    x2 = xt + widths / 2
    y2 = yt + heights / 2

    # 第一個anchor的x1, y1, x2, y2 (有10個)
    # np.hstack((x1, y1, x2, y2)) 的效果是將 x1, y1, x2, y2 這幾個 shape=(N,1) 的列向量拼接成 shape=(N,4) 的矩陣
    # 例如: x1=[[1],[2]], y1=[[3],[4]], x2=[[5],[6]], y2=[[7],[8]]
    # np.hstack((x1, y1, x2, y2)) -> [[1,3,5,7],[2,4,6,8]]
    base_anchors = np.hstack((x1, y1, x2, y2))

    # 生成所有anchor
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchors + [j, i, j, i])

    anchor = np.array(anchor).reshape((-1, 4))
    # 可視化anchors
    # print('shift_x.shape: ', shift_x.shape, 'shift_y.shape: ', shift_y.shape)
    # print('base_anchors.shape: ', base_anchors.shape)
    # img = np.zeros((w, h, 3), dtype=np.uint8)
    # for i in range(anchor.shape[0]):
    #     cv2.rectangle(img, (int(anchor[i, 0]), int(anchor[i, 1])), (int(anchor[i, 2]), int(anchor[i, 3])), (0, 0, 255), 2)
    #     cv2.imwrite('anchor.jpg', img)
    #     cv2.imshow('anchor', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return anchor


def cal_overlap(base_anchors, gtboxes):
    """
    计算anchors和ground truth boxes之间的IOU
    
    Args:
        base_anchors: [N, 4] anchor boxes (x1, y1, x2, y2)
        gtboxes: [M, 4] ground truth boxes (x1, y1, x2, y2)
    
    Returns:
        overlaps: [N, M] IOU矩阵
    """
    # 计算所有anchor和gt box之间的IOU
    overlaps = np.zeros((base_anchors.shape[0], gtboxes.shape[0]))
    anchor_areas = (base_anchors[:, 2] - base_anchors[:, 0]) * (base_anchors[:, 3] - base_anchors[:, 1])
    
    for k in range(gtboxes.shape[0]):
        gt_box = np.tile(gtboxes[k, :], (base_anchors.shape[0], 1))  # 当前GT box [x1, y1, x2, y2]
        gt_area = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
        # 计算交集区域的坐标
        # 交集左上角：max(anchor_x1, gt_x1), max(anchor_y1, gt_y1)
        # 交集右下角：min(anchor_x2, gt_x2), min(anchor_y2, gt_y2)
        inter_x1 = np.maximum(base_anchors[:, 0], gt_box[:, 0])
        inter_y1 = np.maximum(base_anchors[:, 1], gt_box[:, 1])
        inter_x2 = np.minimum(base_anchors[:, 2], gt_box[:, 2])
        inter_y2 = np.minimum(base_anchors[:, 3], gt_box[:, 3])

        # 计算交集面积
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        
        # 计算并集面积
        union = anchor_areas + gt_area - intersection
        # 计算IOU，避免除零
        overlaps[:, k] = intersection / (union)
    
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
    # 从边界坐标计算中心点坐标和宽高
    # Anchor boxes的中心点和宽高
    anchor_yt = (base_anchors[:, 3] + base_anchors[:, 1]) / 2
    anchor_ht = base_anchors[:, 3] - base_anchors[:, 1] + 1
    
    # Ground truth boxes的中心点和宽高
    gt_yt = (gtboxes[:, 3] + gtboxes[:, 1]) / 2
    gt_ht = gtboxes[:, 3] - gtboxes[:, 1] + 1
    
    # 计算回归目标
    dy = (gt_yt - anchor_yt) / anchor_ht
    dh = np.log(gt_ht / anchor_ht)
    
    # transpose的作用是將原本 shape 為 (2, N) 的數組轉置為 (N, 2)，即每一行對應一個anchor的(dy, dh)回歸目標
    return np.vstack((dy, dh)).transpose()


def cal_rpn(img_size, feature_size, scale, gtboxes):
    h, w = img_size

    print('img_size: ', img_size, '\n', 'feature_size: ', feature_size, '\n', 'scale: ', scale)
    # gen base anchors
    base_anchors = gen_anchors(feature_size, scale)
    
    # calculate iou between gtboxes and base_anchors
    overlaps = cal_overlap(base_anchors, gtboxes)
    print('overlaps.shape: ', overlaps.shape)
    print('gtboxes.shape: ', gtboxes.shape)
    

    # 获取每个gtbox与anchor的最大iou
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    print('anchor_argmax_overlaps.shape: ', anchor_argmax_overlaps.shape)
    # 获取每个anchor与gtbox的最大iou
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # 初始化labels -1 don't care, 0 background, 1 foreground
    labels = np.empty(base_anchors.shape[0])
    labels.fill(-1)

    # 确保每个gtbox至少有一个前景anchor
    labels[gt_argmax_overlaps] = 1
    # 如果anchor与gtbox的iou > IOU_POSITIVE, label = 1
    labels[anchor_max_overlaps >= config.IOU_POSITIVE] = 1
    # 如果anchor与gtbox的iou < IOU_NEGATIVE, label = 0
    labels[anchor_max_overlaps < config.IOU_NEGATIVE] = 0
    
    # 检查anchor是否在图像之外
    outside_anchor = np.where(
        (base_anchors[:, 0] < 0) |  
        (base_anchors[:, 1] < 0) |
        (base_anchors[:, 2] > w) |
        (base_anchors[:, 3] > h)
    )[0]
    labels[outside_anchor] = -1

    # 打印正負樣本和忽略樣本的數量
    num_pos = np.sum(labels == 1)
    num_neg = np.sum(labels == 0)
    num_ign = np.sum(labels == -1)
    print(f'正樣本數比例: {num_pos/len(labels)}, 負樣本比例: {num_neg/len(labels)}, 忽略樣本比例: {num_ign/len(labels)}')

    # 如果前景anchor数量大于RPN_FORE_NUM(默认128), 则随机选择RPN_FORE_NUM个前景anchor
    fg_index = np.where(labels == 1)[0]
    if len(fg_index) > config.RPN_FORE_NUM:
        labels[np.random.choice(fg_index, len(fg_index) - config.RPN_FORE_NUM, replace=False)] = -1

    # gtboxes.shape == (102, 4), anchor_argmax_overlaps.shape == (75000,)
    # 這裏沒有報錯是因為 anchor_argmax_overlaps 是每個 anchor 對應的最大 iou 的 gtbox 索引
    # 所以 gtboxes[anchor_argmax_overlaps, :] 會返回一個 (75000, 4) 的數組

    # 每個 anchor 都有一個對應的 gtbox 坐標
    print('gtboxes[anchor_argmax_overlaps, :].shape: ', gtboxes[anchor_argmax_overlaps, :].shape)
    bbox_targets = bbox_transform(base_anchors, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets], base_anchors

'''
已知 anchor和差异参数 regression_factor(Vc, Vh),计算目标框 bbox
'''
def transform_bbox(anchor, regression_factor):
    anchor_y = (anchor[:, 1] + anchor[:, 3]) * 0.5
    anchor_x = (anchor[:, 0] + anchor[:, 2]) * 0.5
    anchor_h = anchor[:, 3] - anchor[:, 1] + 1

    Vc = regression_factor[0, :, 0]
    Vh = regression_factor[0, :, 1]

    bbox_y = Vc * anchor_h + anchor_y
    bbox_h = np.exp(Vh) * anchor_h

    x1 = anchor_x - 16 * 0.5
    y1 = bbox_y - bbox_h * 0.5
    x2 = anchor_x + 16 * 0.5
    y2 = bbox_y + bbox_h * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox

'''
bbox 边界裁剪
    x1 >= 0
    y1 >= 0
    x2 < im_shape[1]
    y2 < im_shape[0]
'''
def clip_box(bbox, im_shape):
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox

'''
bbox尺寸过滤，舍弃小于设定最小尺寸的bbox
'''
def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep

# 非極大值抑制
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

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
        # print("根据索引返回一个样本")

        img_name = self.img_list[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)

        h, w, c = img.shape

        # 缩放图片至1600
        rescale_fac = max(h, w) / 1000
        if rescale_fac > 1:
            h = int(h/rescale_fac)
            w = int(w/rescale_fac)
            # 縮放gtbox
            img = cv2.resize(img, (w, h))

        # 读取标注文件
        gt_path = os.path.join(self.gt_dir, 'gt_'+img_name)
        print('gt_path: ', gt_path)
        gtbox = self.parse_gtfile(gt_path, rescale_fac)

        # 將gtbox的每個頂點坐標標簽依次連接, 變成一個矩形框, 並繪製出來
        gtbox = np.array(gtbox)
        # for i in range(gtbox.shape[0]):
        #     cv2.rectangle(img, (int(gtbox[i, 0]), int(gtbox[i, 1])), (int(gtbox[i, 2]), int(gtbox[i, 3])), (0, 0, 255), 2)
        # cv2.imwrite('gtbox.jpg', img)
        # cv2.imshow('gtbox', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 水平翻转
        rd = np.random.random()
        if rd < 0.5:
            # 水平翻转, 參數1
            img = cv2.flip(img, 1)
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        # # 垂直翻转
        # if rd > 0.3 and rd < 0.6:
        #     # 垂直翻转, 參數0
        #     img = cv2.flip(img, 0)
        #     newy1 = h - gtbox[:, 3] - 1
        #     newy2 = h - gtbox[:, 1] - 1
        #     gtbox[:, 1] = newy1
        #     gtbox[:, 3] = newy2

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
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        regr = torch.from_numpy(regr).float()
        cls = torch.from_numpy(cls).float()

        return m_img, cls, regr

    # 將四個頂點坐標轉換為矩形框坐標
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
            # 爲了切分出整數個anchor，所以需要//16 + 1
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16*i-0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
                
            gtboxes.append((prev, ymin, xmax, ymax))
        print('len(gtboxes): ', len(gtboxes))
        return np.array(gtboxes)
    
    # 解析标注文件
    def parse_gtfile(self, gt_path, rescale_fac):
        if '.jpg' in gt_path:
            gt_path = gt_path.replace('.jpg', '.txt')
        if '.png' in gt_path:
            gt_path = gt_path.replace('.png', '.txt')
        if '.gif' in gt_path:
            gt_path = gt_path.replace('.gif', '.txt')

        # print('rescale_fac: ', rescale_fac)
        # print('gt_path: ', gt_path)

        with open(gt_path, 'r', encoding="utf-8-sig") as f:
            gtbox = []
            gt = f.readlines()
            for line in gt:
                # 切割成8個數字
                line = line.split(',')[:8]
                # 將數字轉換為float
                x1, y1, x2, y2, x3, y3, x4, y4 = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7])
                gtbox.append([x1, y1, x2, y2, x3, y3, x4, y4])
            
        gtbox = np.array(gtbox)
        gtbox = self.box_transfer_v2(gtbox, rescale_fac)


        return gtbox


    
    








