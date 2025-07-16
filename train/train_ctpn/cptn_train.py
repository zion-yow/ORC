import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

import config
from data.dataset import ICDARDataset

random_seed = 2025
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

epochs = 30
lr = 1e-3
resume_epoch = 0

def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    check_path = os.path.join(
        config.checkpoints_dir,
        f'v3_ctpn_ep{epoch:02d}_'
        f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}'
         )

    torch.save(state, check_path)
    print(f'Checkpoint saved to {check_path}')




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if os.path.exists(config.checkpoints_dir):
        # checkpoint = torch.load(os.path.join(config.checkpoints_dir, 'v3_ctpn_ep00_0.0000_0.0000_0.0000.pth'))
        # print(f'Loading checkpoint from {config.checkpoints_dir}')
    # else:
    checkpoint = False
    pretrained = False

    dataset = ICDARDataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.workers)
    model = CTPN_Model()
    model.to(device)

    if checkpoint:
        # print(f'Loading checkpoint from {config.checkpoints_dir}')
        # cc = torch.load(os.path.join(config.checkpoints_dir, 'v3_ctpn_ep00_0.0000_0.0000_0.0000.pth'))
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
        print(f'Resuming training from epoch {resume_epoch}')
    else:
        # model.apply(weights_init)
        pass

