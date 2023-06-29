# -*- coding:utf-8 -*-
#@Project: SOSMaskFuse for image fusion
#@Author: Qian Xuanhu, Shenzhen University
#@Email: qianxuanhu@163.com
#@File : SOSnetwork_test.py

import os
from torch.utils.data import DataLoader
from SOSnetwork.lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from SOSnetwork.HRNet import HRNet
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def SOSMask_run(test_path, out_path_root, SOSnetwork_model_path):
    model_path = SOSnetwork_model_path
    if not os.path.exists(out_path_root):
        os.mkdir(out_path_root)
    data = Data(root=test_path, mode='test')
    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = HRNet(mode='test').cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))

    out_path = os.path.join(out_path_root, 'SOSmask')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    time_s = time.time()
    img_num = len(loader)
    net.eval()
    with torch.no_grad():
        for inf, _, (H, W), name in loader:
            print(os.path.join(out_path, name[0]))
            score = net(inf.cuda().float())
            score = F.interpolate(score, size=(H, W), mode='bilinear', align_corners=True)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            multi_fuse = pred * 255
            out_img_path = os.path.join(out_path, name[0])
            cv2.imwrite(out_img_path, multi_fuse)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))

if __name__ == '__main__':
    test_path = '../Test_images/InfVis/'
    out_path_root = './output'
    SOSnetwork_model_path = './model/Final_normal.pth'
    SOSMask_run(test_path, out_path_root, SOSnetwork_model_path)


