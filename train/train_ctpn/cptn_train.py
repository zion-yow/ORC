#-*- coding:utf-8 -*-
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from data.dataset import ICDARDataset
import config
import visdom

random_seed = 1
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

epochs = 80
lr = 1e-3
# 已經訓練過幾個epoch
resume_epoch = 0


def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    check_path = os.path.join(config.checkpoints_dir, 'ctpn_ep{:02d}_{:.4f}_{:.4f}_{:.4f}.'.format(epoch, loss_cls, loss_regr, loss) + ext)
    try:
        torch.save(state, check_path)
    except BaseException as e:
        print(e)
        print('fail to save to {}'.format(check_path))
    print('saving to {}'.format(check_path))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    dataset = ICDARDataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    
    model = CTPN_Model().to(device)

    checkpoints_weight = config.pretrained_weights
    print('exist pretrained ',os.path.exists(checkpoints_weight))   
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55, 70, 90, 120], gamma=0.1, last_epoch=-1)

    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)
    
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch

    viz = visdom.Visdom(env='ctpn-train', port=8097)

    if resume_epoch == 0:
        n_iter = 0
    else:
        n_iter = resume_epoch * len(dataset)

    # 一個epoch相當於把所有樣本都過一遍
    
    for epoch in range(resume_epoch, epochs + resume_epoch):
        print('Epoch {}/{}'.format(epoch, epochs + resume_epoch))
        # epoch_size等於數據集的樣本數，因為batch_size=1
        epoch_size = len(dataset)
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)
        for param_group in scheduler.optimizer.param_groups:
            print('lr: %s'% param_group['lr'])
        print('#'*80)

        # dataloader會遍歷整個數據集，每個樣本都會被訓練一次
        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            since = time.time()
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)
    
            optimizer.zero_grad()

            # 前向传播
            # 調用 model(imgs) 時，會執行 CTPN_Model 類中的 forward 方法。
            # 這個方法會將輸入的圖像 imgs 經過 VGG16 backbone 提取特徵，然後通過卷積層、雙向GRU、1x1卷積等結構，
            # 最終輸出兩個張量：out_cls（每個anchor的分類分數）和 out_regr（每個anchor的回歸參數）。
            # 這兩個輸出分別用於文本區域的分類和邊框回歸。
            out_cls, out_regr = model(imgs)

            # 计算损失
            loss_regr = critetion_regr(out_regr, regrs)
            loss_cls = critetion_cls(out_cls, clss)
            loss = loss_cls + loss_regr 

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
    
            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i + 1
            n_iter += 1

            print('time:{}'.format(time.time() - since))
            print(  'EPOCH:{}/{}--BATCH:{}/{}\n'.format(epoch, epochs + resume_epoch, batch_i, epoch_size),
                    'batch: loss_cls:{:.4f}--loss_regr:{:.4f}--loss:{:.4f}\n'.format(loss_cls.item(), loss_regr.item(), loss.item()),
                    'epoch: loss_cls:{:.4f}--loss_regr:{:.4f}--loss:{:.4f}\n'.format(epoch_loss_cls/mmp, epoch_loss_regr/mmp, epoch_loss/mmp)
                )
            if mmp % 100 == 0:
                viz.line(Y=np.array([epoch_loss_cls/mmp]), X=np.array([n_iter//100]), 
                                    update='append', win='loss_cls', opts={'title':'loss_cls'})
                viz.line(Y=np.array([epoch_loss_regr/mmp]), X=np.array([n_iter//100]), 
                                    update='append', win='loss_regr', opts={'title'
                                    :'loss_regr'})
                viz.line(Y=np.array([epoch_loss/mmp]), X=np.array([n_iter//100]), 
                                    update='append', win='loss_all', opts={'title':'loss_all'})
                
            
            # break

        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print('Epoch:{}--{:.4f}--{:.4f}--{:.4f}'.format(epoch, epoch_loss_cls, epoch_loss_regr, epoch_loss))
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(), 'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)
        
        # break
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()