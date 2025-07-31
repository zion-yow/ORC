from torch import nn
from torchvision import models
import torch
# import torch.nn.functional as F
import config

'''
回归损失: smooth L1 Loss
只针对正样本求取回归损失
L = 0.5*x**2  |x|<1
L = |x| - 0.5
sigma: 平滑系数
1、从预测框p和真值框g中筛选出正样本
2、|x| = |g - p|
3、求取loss，这里设置了一个平滑系数 1/sigma
  (1) |x|>1/sigma: loss = |x| - 0.5/sigma
  (2) |x|<1/sigma: loss = 0.5*sigma*|x|**2
'''
class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device
    
    def forward(self, input, target):
        try:
            # 因爲target是(batch, nums_of_anchor, 3)，所以target[0, :, 0]是第0維的第0個元素，即cls
            cls = target[0, :, 0]
            regression = target[0, :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regression[regr_keep]

            # 因爲input是(batch, nums_of_anchor, 2)，所以input[0]是第0維的第0個元素，即regr
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)

            # 計算loss
            less_one = (diff<1.0/self.sigma).float()
            # 誤差大的時候，loss = |x| - 0.5/sigma
            # 誤差小的時候，loss = 0.5*sigma*|x|**2, 對異常值敏感
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            loss = torch.tensor(0.0)

        return loss.to(self.device)

'''
分类损失: softmax loss
1、OHEM模式
  (1) 筛选出正样本，求取softmaxloss
  (2) 求取负样本数量N_neg, 指定样本数量N, 求取负样本的topK loss, 其中K = min(N_neg, N - len(pos_num))
  (3) loss = loss1 + loss2
2、求取NLLLoss，截断在(0, 10)区间
'''
class RPN_CLS_Loss(nn.Module):
    def __init__(self,device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        if config.OHEM:
            
            # 這裏應該是取出所有 anchor 的標籤 (cls)，target 形狀是 (batch, nums_of_anchor, 3)
            # target[0, :, 0] 代表第0個 batch 的所有 anchor 的分類標籤+
            print('target:', target)
            print('target.shape:', target.shape)
            cls_gt = target[0][0]
            num_pos = 0
            loss_pos_sum = 0

            # 如果正樣本數量不為0，則計算正樣本的損失
            if len((cls_gt == 1).nonzero()) != 0: 
                # 獲取正樣本的索引
                cls_pos = (cls_gt == 1).nonzero()[:, 0]
                # 獲取正樣本的標籤
                gt_pos = cls_gt[cls_pos].long()
                # 獲取正樣本的預測分數
                cls_pred_pos = input[0][cls_pos]
                # 計算正樣本的損失
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)

            # 獲取負樣本的索引
            cls_neg = (cls_gt == 0).nonzero()[:, 0]
            # 獲取負樣本的標籤
            gt_neg = cls_gt[cls_neg].long()
            # 獲取負樣本的預測分數
            cls_pred_neg = input[0][cls_neg]

            # 計算負樣本的損失
            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            # torch.topk的作用是從損失中選出最大的K個，這裡是選出損失最大的負樣本（最難分的負樣本），用於OHEM（Online Hard Example Mining）
            # 這樣可以讓模型更關注難以分類的負樣本，提高訓練效果
            K = min(len(loss_neg), config.RPN_TOTAL_NUM - num_pos)
            loss_neg_topK, _ = torch.topk(loss_neg, K)
            loss_cls = loss_pos_sum + loss_neg_topK.sum()
            loss_cls = loss_cls / config.RPN_TOTAL_NUM

            print('num_pos:', num_pos, 'num_neg:', len(cls_neg))
            # print('loss_neg_topK:', loss_neg_topK)
            # print('cls_pred_pos:', cls_pred_pos)
            # print('cls_pred_neg:', cls_pred_neg)
        
            return loss_cls.to(self.device)
        else:
            y_true = target[0][0]
            cls_keep = (y_true != -1).nonzero()[:, 0]
            cls_true = y_true[cls_keep].long()
            cls_pred = input[0][cls_keep]
            loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
            loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)

            return loss.to(self.device)




class basic_conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(basic_conv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # 這裡的bn是batch normalization，用於加速訓練和穩定模型
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(weights=None)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(128 * 2, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)


    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)
        # permute會改變tensor的維度順序，這裏x.permute(0, 2, 1)是把x的第2和第3個維度交換
        x1 = x.permute(0, 2, 3, 1)

        b = x1.size()
        # .view的功能是重新排列tensor的形狀，這裡將x1從(b[0], b[1], b[2], b[3])變成(b[0]*b[1], b[2], b[3])
        x1 = x1.view(b[0]*b[1], b[2], b[3])

        x2, _ = self.brnn(x1)

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)

        # contiguous() 的作用是返回一個內存連續的tensor副本，這在某些操作（如view）之後是必需的
        x3 = x3.permute(0, 3, 1, 2).contiguous()

        x3 = self.lstm_fc(x3)
        
        x = x3
        cls = self.rpn_class(x)

        regr = self.rpn_regress(x)

        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()

        cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2)

        regr = regr.view(regr.size(0), regr.size(1)*regr.size(2)*10, 2)

        return cls, regr

