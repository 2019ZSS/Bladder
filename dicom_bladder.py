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


def make_dataset_dcm(root, mode, is_dcm=True):
    assert mode in ['train', 'val']
    items = []
    if mode == 'train':
        if is_dcm:
            img_path = os.path.join(root, 'Dicom')
        else:
            img_path = os.path.join(root, 'Dicom2png')
        mask_path = os.path.join(root, 'Label(Predicted)')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'train.txt')).readlines()]
        for x in data_list:
            item  = []
            if is_dcm:
                tmp = x.split('_')
                item.append(os.path.join(img_path, tmp[0], tmp[1] + '.dcm'))
            else:
                item.append(os.path.join(img_path, x + '.png')) 
            item.append(os.path.join(mask_path, x + '.png'))
            items.append(item)

    elif mode == 'val':
        if is_dcm:
            img_path = os.path.join(root, 'Dicom')
        else:
            img_path = os.path.join(root, 'Dicom2png')
        mask_path = os.path.join(root, 'Label(Predicted)')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'val.txt')).readlines()]
        for x in data_list:
            item  = []
            if is_dcm:
                tmp = x.split('_')
                item.append(os.path.join(img_path, tmp[0], tmp[1] + '.dcm'))
            else:
                item.append(os.path.join(img_path, x + '.png')) 
            item.append(os.path.join(mask_path, x + '.png'))
            items.append(item)
    return items



class Bladder(data.Dataset):

    def __init__(self, root, mode, is_dcm=True, make_dataset_fn=None, joint_transform=None, transform=None, target_transform=None):
        if make_dataset_fn is None:
            self.imgs = make_dataset(root, mode)
        else:
            self.imgs = make_dataset_fn(root, mode, is_dcm)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        if not is_dicom_file(img_path):
            img = Image.open(img_path)
            img = np.array(img)
        else:
            img = dcm_to_img_numpy(img_path)
        
        mask = Image.open(mask_path)
        mask = np.array(mask)


        # 输入要求为numpy数组
        if self.joint_transform is not None:
            img = self.joint_transform(img)
            mask = self.joint_transform(mask)

        # 输出为PIL图像类型
        img = np.array(img)
        mask = np.array(mask)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, self.palette)
        # shape from (H, W, C) to (C, H, W)
        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

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
    target_transform = extended_transforms.MaskToTensor()
    make_dataset_fn = make_dataset_dcm
    root = './hospital_data/MRI_T2'
    mode = 'val'
    batch_size = 2
    dataset = Bladder(root, mode, make_dataset_fn=make_dataset_fn, 
                        joint_transform=joint_transform, transform=transform, target_transform=target_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for x in data_loader:
        print(type(x))
        # break 
