#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    batch_count = pred.shape[0]
    point_count = pred.shape[1]
    cls_count = pred.shape[2]

    pred = pred.permute((0,2,1)).reshape((batch_count,cls_count,point_count,1))#torch.Size([32, 40, 2048])
    gold = gold.reshape((batch_count,point_count,1))


    loss = F.cross_entropy(pred, gold)

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
