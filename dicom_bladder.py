# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# dicom_bladder.py
# @description: ...
'''
import os
import cv2
import torch
import numpy as np
import utils.transforms as extended_transforms
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import helpers
from torch.utils.data import DataLoader
from dicom_data import is_dicom_file, dcm_to_img_numpy


'''
128 = bladder
255 = tumor
0 = background 
'''
palette = [[0], [128], [255]]
num_classes = 3


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        pass
    elif mode == 'val':
        pass
    elif mode == 'test':
        files = os.listdir(root)
        for file in files:
            items.append(os.path.join(root, file))
    return items


def make_dataset_dcm(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        pass
    elif mode == 'val':
        pass
    elif mode == 'test':
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'test.txt')).readlines()]
        for item in data_list:
            items.append(item)
    return items


class Bladder(data.Dataset):

    def __init__(self, root, mode, make_dataset_fn=None, joint_transform=None, transform=None):
        if make_dataset_fn is None:
            self.imgs = make_dataset(root, mode)
        else:
            self.imgs = make_dataset_fn(root, mode)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]

        if not is_dicom_file(img_path):
            img = Image.open(img_path)
            img = np.array(img)
        else:
            img = dcm_to_img_numpy(img_path)
        
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # 输入要求为numpy数组
        if self.joint_transform is not None:
            img = self.joint_transform(img)

        # 输出为PIL图像类型
        img = np.array(img)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        img = np.expand_dims(img, axis=2)
        # shape from (H, W, C) to (C, H, W)
        img = img.transpose([2, 0, 1])
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)
        

if __name__ == "__main__":
    # make_dataset(root, mode)
    joint_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(128)
    ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform = extended_transforms.ImgToTensor()
    make_dataset_fn = make_dataset_dcm
    root = './hospital_data/dicom_to_png'
    root = './hospital_data/3d'
    mode = 'test'
    batch_size = 2
    dataset = Bladder(root, mode, make_dataset_fn=make_dataset_fn, 
                        joint_transform=joint_transform, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for x in data_loader:
        print(type(x))
        print(x.size())
        print(x)
        break 
