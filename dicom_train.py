# !/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @description: ...
'''
import time
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

import dicom_bladder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2
n_epoch = 30
model_name = 'U_Net_'
loss_name = 'bcew_'
times = 'no_' + str(n_epoch)
extra_description = '3d'
writer = SummaryWriter(os.path.join('./log/bladder_trainlog',  'bladder_exp', model_name+loss_name+times+extra_description))
resume = False
data_path = './hospital_data/MRI_T2'


joint_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(128)
])
transform = extended_transforms.ImgToTensor()
target_transform = extended_transforms.MaskToTensor()
make_dataset_fn = dicom_bladder.make_dataset_dcm


train_set = dicom_bladder.Bladder(data_path, 'train', is_dcm=True,
                        make_dataset_fn=make_dataset_fn, joint_transform=joint_transform, 
                        transform=transform, target_transform=target_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

net = U_Net(img_ch=1, num_classes=3).to(device)

if loss_name == 'dice_':
    criterion = SoftDiceLossV2(activation='sigmoid', num_classes=3).to(device)
elif loss_name == 'bcew_':
    criterion = nn.BCEWithLogitsLoss().to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-4)

modle_2d = './model/checkpoint/exp/U_Net_bcew_no_30.pth'
checkpoint = torch.load(modle_2d)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train(train_loader, net, criterion, optimizer, num_epoches , iters):
    if resume:
        CHECKPOINT_FILE = './model/checkpoint/exp/{}.pth'.format(model_name + loss_name + times + extra_description)
        # 恢复上次的训练状态
        print("Resume from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch'] + 1
    else:
        initepoch = 1
    for epoch in range(initepoch, num_epoches + 1):
        try:
            # 开始时间
            st = time.time()
            b_dice = 0.0
            t_dice = 0.0
            d_len = 0
            # 开始训练
            for inputs, mask in train_loader:
                X = inputs.to(device)
                y = mask.to(device)
                optimizer.zero_grad()
                output = net(X)
                loss = criterion(output, y)
                output = torch.sigmoid(output)
                output[output < 0.5] = 0
                output[output > 0.5] = 1
                bladder_dice = diceCoeffv2(output[:, 0:1, :], y[:, 0:1, :], activation=None).cpu().item()
                tumor_dice = diceCoeffv2(output[:, 1:2, :], y[:, 1:2, :], activation=None).cpu().item()
                mean_dice = (bladder_dice + tumor_dice) / 2
                d_len += 1
                b_dice += bladder_dice
                t_dice += tumor_dice
                loss.backward()
                optimizer.step()
                iters += batch_size
                string_print = "Epoch = %d iters = %d Current_Loss = %.4f Mean Dice=%.4f Bladder Dice=%.4f Tumor Dice=%.4f Time = %.2f"\
                            % (epoch, iters, loss.item(), mean_dice,
                                bladder_dice, tumor_dice, time.time() - st)
                tools.log(string_print)
                st = time.time()
                writer.add_scalar('train_main_loss', loss.item(), iters)

            b_dice = b_dice / d_len
            t_dice = t_dice / d_len
            m_dice = (b_dice + t_dice) / 2

            print('Epoch {}/{},Train Mean Dice {:.4}, Bladder Dice {:.4}, Tumor Dice {:.4}'.format(
                epoch, num_epoches, m_dice, b_dice, t_dice
            ))
            if epoch == num_epoches:
                checkpoint = {
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }
                torch.save(checkpoint, './model/checkpoint/exp/{}.pth'.format(model_name + loss_name + times + extra_description))
                writer.close()

        except BaseException as e:
            print(e)
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, './model/checkpoint/exp/{}.pth'.format(model_name + loss_name + times + extra_description))
            writer.close()
            print('训练停止')
            return


if __name__ == '__main__':
    train(train_loader, net, criterion, optimizer, n_epoch, 0)