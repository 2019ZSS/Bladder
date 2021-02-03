# baldder.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import helpers

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
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        if 'Augdata' in root:
            data_list = os.listdir(os.path.join(root, 'Images'))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        pass
    return items


def make_dataset_v2(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'Image(split_by_people&class)')
        mask_path = os.path.join(root, 'Label(split_by_people&class)')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'Image(split_by_people&class)')
        mask_path = os.path.join(root, 'Label(split_by_people&class)')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        pass
    return items


class Bladder(data.Dataset):
    def __init__(self, root, mode, joint_transform=None, center_crop=None, transform=None, target_transform=None, make_dataset_fn=None):
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
        self.center_crop = center_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.center_crop is not None:
            img, mask = self.center_crop(img, mask)
        img = np.array(img)
        mask = np.array(mask)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
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
    print(233)
    items = make_dataset('./data', 'train')
    print(items)
    train_set = Bladder('./data', 'train')