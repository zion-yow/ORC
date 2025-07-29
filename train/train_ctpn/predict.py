import torch
import os
import cv2
import config
import torch.nn.functional as F
from ctpn_model import CTPN_Model
import numpy as np
from data.dataset import gen_anchors, transform_bbox, clip_box, filter_bbox, nms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = config.pretrained_weights
model = CTPN_Model().to(device)
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.eval()


def get_text_boxes(image, prob_thresh=0.5, display=True):
    h, w= image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w,h))
        h, w = image.shape[:2]
    image_c = image.copy()
    image = image.astype(np.float32) - [123.68, 116.78, 103.94]
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        cls, regr = model(image)
        # print('cls: ', cls)
        # print('regr: ', regr)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()


        anchors = gen_anchors((int(h / 16), int(w / 16)), 16)

        # 偏移bbox
        bbox = transform_bbox(anchors, regr)
        bbox= clip_box(bbox, (h, w))

        fg = np.where(cls_prob[0, :, 0] > prob_thresh)[0]

        print('cls_prob: ', cls_prob)
        print('正樣本數量: ', np.sum(cls_prob[0, :, 0] > prob_thresh))
        print('fg: ', fg)

        print('bbox.shape: ', bbox.shape)
        print('bbox: ', bbox)


        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)

        # print('select_anchor.shape: ', select_anchor.shape)
        # print('select_anchor: ', select_anchor)
        # print('select_score.shape: ', select_score.shape)
        # print('select_score: ', select_score)

        keep_index = filter_bbox(select_anchor, 16)

        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        

        if display:
            for i in select_anchor:
                # s = str(round(i[-1] * 100, 2)) + '%'
                # i = [int(j) for j in i]
                cv2.rectangle(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)

                # cv2.putText(image_c, s, (i[0]+13, i[1]+13), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

        return select_anchor, image_c


if __name__ == '__main__':
    
    checkpoint = torch.load(weights, map_location=device)
    img_path = r'F:\projects\OCR\test.jpg'
    input_img = cv2.imread(img_path)
    select_anchor, out_img = get_text_boxes(input_img)
    cv2.imwrite('test_results.jpg', out_img)




