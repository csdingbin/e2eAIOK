#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/15/2022 4:17 PM

from torch.utils.data import Dataset
from .utils import rgb_loader,l_loader
import logging
import os

class Office31(Dataset):
    ''' Office31 Dataset

    '''
    def __init__(self, data_path, label_map,data_transform=None, img_mode='RGB'):
        ''' Init method

        :param data_path: img data location
        :param label_map: map from label name to label id
        :param data_transform: data tramsform
        :param img_mode: img mode
        '''
        self.data_path = data_path
        self._label_map = label_map
        self.data_transform = data_transform
        self.img_mode = img_mode
        self._getFileAndImgPath(data_path)

        if data_transform is not None:
            logging.debug("Applying data_transform on %s" % data_path)

        if img_mode.upper() == 'RGB':
            self.loader = rgb_loader
        elif img_mode.upper()  == 'L':
            self.loader = l_loader
        else:
            logging.error("mode muse be one of: 'RGB','L' (ignore letter case), but found :%s"%img_mode)
            raise ValueError("mode muse be one of: 'RGB','L' (ignore letter case), but found :%s"%img_mode)

    def _getFileAndImgPath(self,root_path):
        self.imgs = []

        for label_name in sorted(os.listdir(root_path)):
            label_path = "%s/%s"%(root_path,label_name)
            label_id = self._label_map[label_name]
            for img_name in os.listdir(label_path):
                img_path = "%s/%s"%(label_path,img_name)
                self.imgs.append((img_path,label_id))

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.data_transform is not None:
            img = self.data_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def __str__(self):
        return 'ImageList: image num [%s], data_transform [%s], img_mode [%s]'%(
            len(self.imgs),self.data_transform,self.img_mode
        )
