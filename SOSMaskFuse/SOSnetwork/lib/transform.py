#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, inf, mask):
        for op in self.ops:
            inf, mask = op(inf, mask)
        return inf, mask



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, inf, mask):
        inf = (inf - self.mean) / self.std
        # inf /= 255
        mask /= 255
        return inf, mask

class Minusmean(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, inf, mask):
        inf = inf - self.mean
        mask /= 255
        return inf, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, inf, mask):
        inf = cv2.resize(inf, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return inf, mask

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, inf, mask):
        H, W, _ = inf.shape
        xmin = np.random.randint(W-self.W+1)
        ymin = np.random.randint(H-self.H+1)
        inf = inf[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return inf, mask

class RandomHorizontalFlip(object):
    def __call__(self, inf, mask):
        if np.random.randint(2) == 1:
            inf = inf[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        return inf, mask

class ToTensor(object):
    def __call__(self, inf, mask):
        inf = torch.from_numpy(inf)
        inf = inf[:, :, 0] * 0.11 + inf[:, :, 1] * 0.59 + inf[:, :, 2] * 0.3
        inf = inf.unsqueeze(0)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        return inf, mask.mean(dim=0, keepdim=True)