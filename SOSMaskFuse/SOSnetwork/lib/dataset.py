#coding=utf-8

import os
import cv2
import numpy as np
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset

#BGR
mean_rgb = np.array([[[0.551 * 255, 0.619 * 255, 0.532 * 255]]])
mean_inf = np.array([[[0.341 * 255,  0.360 * 255, 0.753 * 255]]])
std_rgb = np.array([[[0.241 * 255, 0.236 * 255, 0.244 * 255]]])
std_inf = np.array([[[0.208 * 255, 0.269 * 255, 0.241 * 255]]])


class Data(Dataset):
    def __init__(self, root, mode='train'):
        if mode == 'train':
            self.samples = []
            lines = os.listdir(os.path.join(root, 'Inf'))
            for line in lines:
                inf_path = root + '/' + 'Inf' + '/' + line       #os.path.join(root, 'Inf', line)
                maskpath = root + '/' + 'GT' + '/' + line        #os.path.join(root, 'GT', line)
                self.samples.append([inf_path, maskpath])

            self.transform = transform.Compose(transform.Normalize(mean=mean_inf, std=std_inf),
                                               transform.Resize(480, 640),
                                               # transform.RandomHorizontalFlip(),# if need
                                               transform.ToTensor())

        elif mode == 'test':
            self.samples = []
            lines = os.listdir(os.path.join(root, 'Inf'))
            for line in lines:
                inf_path = root + '/' + 'Inf' + '/' + line
                maskpath = root + '/' + 'Inf' + '/' + line  # The mask does not participate when test. Just keep the data format.
                self.samples.append([inf_path, maskpath])

            self.transform = transform.Compose(transform.Normalize(mean=mean_inf, std=std_inf),
                                               transform.Resize(480, 640),
                                               transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        inf_path, maskpath = self.samples[idx]
        inf = cv2.imread(inf_path).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        inf, mask = self.transform(inf, mask)
        return inf, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)