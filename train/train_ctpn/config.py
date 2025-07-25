
base_dir = r'F:\projects\OpenImages'

icdar17_mlt_img_dir = r'F:\projects\OpenImages\OCR\train_data\train_img'
icdar17_mlt_gt_dir = r'F:\projects\OpenImages\OCR\train_data\train_label'

checkpoints_dir = r'F:\projects\OCR\checkpoints'
pretrained_weights = './checkpoints/ctpn.pth'

train_batch_size = 16
test_batch_size = 16
num_workers = 4
epochs = 30
lr = 1e-3
resume_epoch = 0

IOU_POSITIVE = 0.7
IOU_NEGATIVE = 0.3
RPN_FORE_NUM = 150
RPN_TOTAL_NUM = 300

OHEM = True





