# -*- coding:utf-8 -*-
#@Project: SOSMaskFuse for image fusion
#@Author: Qian Xuanhu, Shenzhen University
#@Email: qianxuanhu@163.com
#@File : SOSnetwork_train.py

import os
import torch
import random
import numpy as np
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
from HRNet import HRNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def SOSNet_loss(score, label):

    sal_loss = F.binary_cross_entropy_with_logits(score, label, reduction='mean')

    return sal_loss


if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)

    # dataset
    img_root = './data/road' # train_dataset
    save_path = './model'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    lr = 0.001  # 2
    batch_size = 4
    epoch = 200
    lr_dec = [100, 150]
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)

    net = HRNet(mode='train').cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    iter_num = len(loader)
    net.train()
    all_time = 0
    for epoch_i in range(1, epoch + 1):
        time_s = time.time()
        if epoch_i in lr_dec:
            lr = lr / 10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,
                                  momentum=0.9)
            print(lr)

        prefetcher = DataPrefetcher(loader)
        inf, label = prefetcher.next()
        all_sos_loss = 0
        net.zero_grad()
        i = 0
        while inf is not None:
            i += 1
            score = net(inf)

            sos_loss = SOSNet_loss(score, label)

            all_sos_loss += sos_loss.data
            sos_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 25 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                    epoch_i, epoch, i, iter_num, all_sos_loss / 25))
                all_sos_loss = 0
            inf, label = prefetcher.next()
        if epoch_i % 10 == 0:
            torch.save(net.state_dict(), '%s/normal_epoch_%s.pth' % (save_path, str(epoch_i)))
        time_e = time.time()
        epoch_time = time_e - time_s
        all_time += epoch_time
        print('epoch_time: %f min || time: %f h ' % ((int(epoch_time) / 60), (int(all_time) / 3600)))
    torch.save(net.state_dict(), '%s/Final_normal.pth' % (save_path))