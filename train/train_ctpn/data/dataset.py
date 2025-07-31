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

    print(f'正樣本數比例: {len(np.where(labels == 1)[0])}, 負樣本比例: {len(np.where(labels == 0)[0])}, 忽略樣本比例: {len(np.where(labels == -1)[0])}')
    
    # 检查anchor是否在图像之外
    outside_anchor = np.where(
        (base_anchors[:, 0] < 0) |  
        (base_anchors[:, 1] < 0) |
        (base_anchors[:, 2] > w) |
        (base_anchors[:, 3] > h)
    )[0]
    labels[outside_anchor] = -1
    
    # 如果前景anchor数量大于RPN_FORE_NUM(默认128), 则随机选择RPN_FORE_NUM个前景anchor
    fg_index = np.where(labels == 1)[0]
    if len(fg_index) > config.RPN_FORE_NUM:
        labels[np.random.choice(fg_index, len(fg_index) - config.RPN_FORE_NUM, replace=False)] = -1

    # gtboxes.shape == (102, 4), anchor_argmax_overlaps.shape == (75000,)
    # 這裏沒有報錯是因為 anchor_argmax_overlaps 是每個 anchor 對應的最大 iou 的 gtbox 索引
    # 所以 gtboxes[anchor_argmax_overlaps, :] 會返回一個 (75000, 4) 的數組

    bg_index = np.where(labels == 0)[0]
    num_bg = config.RPN_TOTAL_NUM - np.sum(labels == 1)
    if (len(bg_index) > num_bg):
        # print('bgindex:',len(bg_index),'num_bg',num_bg)
        labels[np.random.choice(bg_index, max(len(bg_index) - num_bg, len(bg_index) - 150), replace=False)] = -1

    print(f'正樣本數比例: {len(np.where(labels == 1)[0])}, 負樣本比例: {len(np.where(labels == 0)[0])}, 忽略樣本比例: {len(np.where(labels == -1)[0])}')


    
    # 每個 anchor 都有一個對應的 gtbox 坐標
    # print('gtboxes[anchor_argmax_overlaps, :].shape: ', gtboxes[anchor_argmax_overlaps, :].shape)
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


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    print('order: ', order)

    keep = []
    while order.size > 0:
        # 選擇分數最高的bbox
        i = order[0]
        keep.append(i)
        # 選擇分數最高的bbox和其他bbox的交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 計算交集與面積的比值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 選擇交集小於閾值的bbox，並將其索引+1
        inds = np.where(ovr <= thresh)[0]

        # 這裏+1是因爲inds是基於order[1:]的索引，所以要映射回原始order的索引需要+1
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
        

        # 水平翻转
        rd = np.random.random()
        if rd < 0.5:
            # 水平翻转, 參數1
            img = cv2.flip(img, 1)
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        # 生成RPN标签
        [cls, regr], base_anchors = cal_rpn(
            (h, w),
            (int(h/16), int(w/16)),
            16,
            gtbox
            )
        
        #
        # 把regr轉換成三維張量, 第一維為長度為1
        # _regr = np.expand_dims(regr, axis=0)
        # post_bbox = transform_bbox(base_anchors, _regr)
        # post_bbox = clip_box(post_bbox, (h, w))
        # post_bbox = post_bbox[np.where(cls == 1)[0]]
        # keep_index = filter_bbox(post_bbox, 16)
        # post_bbox = post_bbox[keep_index]


        # for i in range(post_bbox.shape[0]):
        #     cv2.rectangle(img, (int(post_bbox[i, 0]), int(post_bbox[i, 1])), (int(post_bbox[i, 2]), int(post_bbox[i, 3])), (0, 0, 255), 2)
        # cv2.imwrite(f'post_cls_{index}.jpg', img)
        # cv2.imshow(f'post_cls_{index}', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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
    

class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MAX_HORIZONTAL_GAP = 60
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6

'''
基于图的文本行构造算法
子图连接规则，根据图中配对的文本框生成文本行
1、遍历 graph 的行和列，寻找列全为false、行不全为false的行和列，索引号为index 
2、找到 graph 的第 index 行中为true的那项的索引号，加入子图中，并将索引号迭代给index
3、重复步骤2，直到 graph 的第 index 行全部为false
4、重复步骤1、2、3，遍历完graph
返回文本行list[文本框索引]
'''
class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs    
    
class TextProposalGraphBuilder:
    '''
    构建配对的文本框
    '''
    def get_successions(self, index):
        '''
        遍历[x0, x0+MAX_HORIZONTAL_GAP]
        获取指定索引号的后继文本框
        '''
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def get_precursors(self, index):
        '''
        遍历[x0-MAX_HORIZONTAL_GAP， x0]
        获取指定索引号的前驱文本框
        '''
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def is_succession_node(self, index, succession_index):
        '''
        判断是否是配对的文本框
        '''
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True

        return False

    def meet_v_iou(self, index1, index2):
        '''
        判断两个文本框是否满足垂直方向的iou条件
        overlaps_v: 文本框垂直方向的iou计算。 iou_v = inv_y/min(h1, h2)
        size_similarity: 文本框在垂直方向的高度尺寸相似度。 sim = min(h1, h2)/max(h1, h2)
        '''
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
                size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        '''
        根据文本框构建文本框对
        self.heights: 所有文本框的高度
        self.boxes_table: 将文本框根据左上点的x1坐标进行分组
        graph: bool类型的[n, n]数组，表示两个文本框是否配对，n为文本框的个数
            (1) 获取当前文本框Bi的后继文本框
            (2) 选取后继文本框中得分最高的，记为Bj
            (3) 获取Bj的前驱文本框
            (4) 如果Bj的前驱文本框中得分最高的恰好是 Bi，则<Bi, Bj>构成文本框对
        '''
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True

        return Graph(graph)


class TextProposalConnectorOriented:
    """
    连接文本框，构建文本行bbox
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        '''
        将文本框连接起来，按文本行分组
        '''
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        '''
        一元线性函数拟合X，Y，返回y1, y2的坐标值
        '''
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        '''
        根据文本框，构建文本行
        1、将文本框划分成文本行组，每个文本行组内包含符合规则的文本框
        2、处理每个文本行组，将其串成一个大的文本行
            (1) 获取文本行组内的所有文本框 text_line_boxes
            (2) 求取每个组内每个文本框的中心坐标 (X, Y)，最小、最大宽度坐标值 (x0 ,x1)
            (3) 拟合所有中心点直线 z1
            (4) 设置offset为文本框宽度的一半
            (5) 拟合组内所有文本框的左上角点直线，并返回当x取 (x0+offset, x1-offset)时的极作极右y坐标 （lt_y, rt_y）
            (6) 拟合组内所有文本框的左下角点直线，并返回当x取 (x0+offset, x1-offset)时的极作极右y坐标 （lb_y, rb_y）
            (7) 取文本行组内所有框的评分的均值，作为该文本行的分数
            (8) 生成文本行基本数据
        3、生成大文本框
        '''
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size) 
        
        text_lines = np.zeros((len(tp_groups), 8), np.float32)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            z1 = np.polyfit(X, Y, 1) 

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5 

            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9))
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs







