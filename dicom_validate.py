# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @description: ...
'''
import re
import os
import cv2
import torch
import numpy as np
import utils.transforms as extended_transforms
from PIL import Image
from torch import optim
from torch.utils import data
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import helpers
from utils.loss import SoftDiceLoss, SoftDiceLossV2
from utils.metrics import diceCoeff, diceCoeffv2, diceCoeffv3
from utils import tools
from u_net import *
from dicom_data import is_dicom_file, dcm_to_img_numpy
import dicom_bladder 
import dicom_train as train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS = False 
# numpy 高维数组打印不显示...
np.set_printoptions(threshold=9999999)
batch_size = 1
data_path = './hospital_data/MRI_T2'


joint_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(128)
])
transform = extended_transforms.ImgToTensor()
target_transform = extended_transforms.MaskToTensor()
make_dataset_fn = dicom_bladder.make_dataset_dcm

# val_set = dicom_bladder.Bladder(data_path, 'train', is_dcm=True,
#                         make_dataset_fn=make_dataset_fn, joint_transform=joint_transform, 
#                         transform=transform, target_transform=target_transform)
# # val_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model_name = train.model_name
loss_name = train.loss_name
times = train.times
extra_description = train.extra_description
checkpoint = torch.load("./model/checkpoint/exp/{}.pth".format(model_name + loss_name + times + extra_description))
model = U_Net(img_ch=1, num_classes=3).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
if LOSS:
    writer = SummaryWriter(os.path.join('./log/vallog', 'bladder_exp', model_name+loss_name+times+extra_description))

if loss_name == 'dice_':
    criterion = SoftDiceLossV2(activation='sigmoid', num_classes=3).to(device)
elif loss_name == 'bcew_':
    criterion = nn.BCEWithLogitsLoss().to(device)


def val(model, img_path, mask_path):
    if not is_dicom_file(img_path):
        img = Image.open(img_path)
        img = np.array(img)
    else:
        img = dcm_to_img_numpy(img_path)
    
    mask = Image.open(mask_path)
    mask = np.array(mask)

    img = joint_transform(img)
    mask = joint_transform(mask)

    img = np.asarray(img)
    img = np.expand_dims(img, axis=2)
    mri = img
    mask = np.asarray(mask)
    mask = np.expand_dims(mask, axis=2)
 
    gt = np.float32(helpers.mask_to_onehot(mask, dicom_bladder.palette))
    # 用来看gt的像素值
    gt_showval = gt
    gt = np.expand_dims(gt, axis=3)
    gt = gt.transpose([3, 2, 0, 1])
    gt = torch.from_numpy(gt)

    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, axis=3)
    img = img.transpose([3, 0, 1, 2])
    # numpy -> tensor
    img = transform(img)

    img = img.to(device)
    model = model.to(device)
    pred = model(img)

    pred = torch.sigmoid(pred)
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    bladder_dice = diceCoeffv2(pred[:, 0:1, :], gt.to(device)[:, 0:1, :], activation=None)
    tumor_dice = diceCoeffv2(pred[:, 1:2, :], gt.to(device)[:, 1:2, :], activation=None)
    mean_dice = (bladder_dice + tumor_dice) / 2
    acc = accuracy(pred, gt.to(device))
    p = precision(pred, gt.to(device))
    r = recall(pred, gt.to(device))
    print('mean_dice={:.4}, bladder_dice={:.4}, tumor_dice={:.4}, acc={:.4}, p={:.4}, r={:.4}'
          .format(mean_dice.item(), bladder_dice.item(), tumor_dice.item(),
                  acc.item(), p.item(), r.item()))
    pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
    if transform == 'ImgToTensorV2':
        pred = np.uint8(pred * 255)
    # 用来看预测的像素值
    pred_showval = pred
    pred = helpers.onehot_to_mask(pred, bladder.palette)
    # np.uint8()反归一化到[0, 255]
    imgs = np.uint8(np.hstack([mri, pred, mask]))
    cv2.imshow("mri pred gt", imgs)
    cv2.waitKey(0)
    pred_path = re.findall(r'\d+_[a-zA-Z]*\d+.png', mask_path)[-1]
    cv2.imwrite(os.path.join("./hospital_data/3d_pred", pred_path), imgs, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]) 


if __name__ == '__main__':
    print('validate')
    root = data_path
    mode = 'val'
    imgs = make_dataset_fn(root=root, mode=mode, is_dcm=True)
    for img_path, mask_path in imgs:
        val(model, img_path, mask_path)
        break
    