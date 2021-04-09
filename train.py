# train.py

from logging import root
import time
import os
import sys
from typing import Generator

from PIL.ImageOps import scale
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'.'))

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
 
import bladder
from utils.loss import *
from utils.metrics import *
from utils import tools
from utils import helpers
from u_net import *

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='./hospital_data/2d', help='load input data path')
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--interval", type=int, default=10, help="How long to save the model")
parser.add_argument("--k_fold", type=int, default=1, help='k fold training')
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0,1', help="number of cpu threads to use during batch generation")
parser.add_argument("--scale_size", type=int, default=256, help='scale size of input iamge')
parser.add_argument("--crop_size", type=int, default=256, help='crop size of input image')
parser.add_argument("--model_name", type=str, default='U_Net', help='model name')
parser.add_argument("--optimizer_name", type=str, default='Adam', help='optimizer name')
parser.add_argument("--metric", type=str, default='dice_coef', help='evaluation bladder and tumor loss')
parser.add_argument("--loss_name", type=str, default='bcew_', help='loss_name')
parser.add_argument("--LR_INIT", type=float, default=1e-4, help='init learing rate')
parser.add_argument("--extra_description", type=str, default='', help='some extra information about the model')
parser.add_argument("--resume", type=bool, default=False, help='Whether to continue the training model')
parser.add_argument("--checkpoint_file", type=str, default='', help='checkpoint file to resume the model')
parser.add_argument("--LOSS", type=bool, default=False, help='Whether to record validate loss')
opt = parser.parse_args() 
num_workers = opt.n_cpu
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
data_path = opt.data_path
k_fold = opt.k_fold
scale_size = opt.scale_size
crop_size = opt.crop_size
batch_size = opt.batch_size
n_epoch = opt.epochs
interval = opt.interval
model_name = opt.model_name
optimizer_name = opt.optimizer_name
metric = opt.metric
loss_name = opt.loss_name
LR_INIT = opt.LR_INIT
extra_description = opt.extra_description
resume = opt.resume
checkpoint_file = opt.checkpoint_file
LOSS = opt.LOSS
times = 'no_' + str(n_epoch)
writer = SummaryWriter(os.path.join('./log/bladder_trainlog',  'bladder_exp', '_'.join([model_name, optimizer_name, loss_name, times, extra_description])))


def generate_dir(prefix,model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description):
    keys = [model_name, optimizer_name, loss_name, metric, '{}_fold'.format(k_fold),  str(scale_size), extra_description]
    return os.path.join(prefix, '_'.join(keys))

model_saved_dir = generate_dir('./model/checkpoint/exp', model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description)
if not os.path.exists(model_saved_dir):
    os.makedirs(model_saved_dir)

log_saved_dir = generate_dir('./log/training', model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description)
if not os.path.exists(log_saved_dir):
    os.makedirs(log_saved_dir)

log_file = log_saved_dir + times + '.log'
log = tools.Logger(filename=log_file)

# 数据处理
make_dataset_fn = bladder.make_dataset_v2
if k_fold > 1:
    mode_root = os.path.join(data_path, '{}_fold'.format(k_fold))
else:
    mode_root = None

# 设置数据输入转换
train_joint_transform = joint_transforms.Compose([
    joint_transforms.Scale(scale_size),
    joint_transforms.RandomRotate(10),
    joint_transforms.RandomHorizontallyFlip()
])
center_crop = joint_transforms.CenterCrop(crop_size)
train_input_transform = extended_transforms.ImgToTensor()
target_transform = extended_transforms.MaskToTensor()

if loss_name == 'dice_':
    criterion = SoftDiceLossV2(activation='sigmoid', num_classes=3, dice=metric).to(device)
elif loss_name == 'bcew_':
    criterion = nn.BCEWithLogitsLoss().to(device)
else:
    raise NotImplementedError('{} NotImplemented '.format(loss_name))

if metric == 'diceCoeff':
    metric_fn = diceCoeff
elif metric == 'diceCoeffv2':
    metric_fn = diceCoeffv2
elif metric == 'diceCoeffv3':
    metric_fn = diceCoeffv3
elif metric == 'dice_coef':
    metric_fn = dice_coef
else:
    raise NotImplementedError("{} NotImplemented".format(metric))


def get_dataloader(data_path, mode, kth=-1):
    data_set = bladder.Bladder(root=data_path, mode=mode, mode_root=mode_root, kth=kth,
                        joint_transform=train_joint_transform, center_crop=center_crop,
                        transform=train_input_transform, target_transform=target_transform,
                        make_dataset_fn=make_dataset_fn)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


def main():
    net = U_Net(img_ch=1, num_classes=3).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net.cuda())
    kth = 0 if k_fold > 1 else -1
    train_loader = get_dataloader(data_path=data_path, mode='train', kth=kth)

    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=0.0001)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=LR_INIT, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    else:
        raise NotImplementedError('{} NotImplemented '.format(optimizer_name))

    train(train_loader, net, criterion, optimizer, n_epoch, 0)
 

def save_model(net, optimizer, epoch):
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, os.path.join(model_saved_dir, '{}.pth'.format(epoch)))
 

def train(train_loader, net, criterion, optimizer, num_epoches , iters):
    if resume:
        if checkpoint_file is None:
            CHECKPOINT_FILE = os.path.join(model_saved_dir, '{}.pth'.format(num_epoches))
        else:
            CHECKPOINT_FILE = checkpoint_file
        # 恢复上次的训练状态
        log.logger.info("Resume from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch'] + 1
    else:
        initepoch = 1
    for epoch in range(initepoch, num_epoches + 1):
        net.train()
        if k_fold > 1:
            k_th = epoch % k_fold
            train_loader = get_dataloader(data_path=data_path, mode='train', kth=k_th)
            val_loader = get_dataloader(data_path=data_path, mode='val', kth=k_th)
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
                bladder_dice = metric_fn(output[:, 0:1, :], y[:, 0:1, :], activation=None)
                if hasattr(bladder_dice, 'cpu'):
                    bladder_dice = bladder_dice.cpu().item()
                tumor_dice = metric_fn(output[:, 1:2, :], y[:, 1:2, :], activation=None)
                if hasattr(tumor_dice, 'cpu'):
                    tumor_dice = tumor_dice.cpu().item()
                mean_dice = (bladder_dice + tumor_dice) / 2
                d_len += 1
                b_dice += bladder_dice
                t_dice += tumor_dice
                loss.backward()
                optimizer.step()
                iters += batch_size
                string_print = "Epoch = %d iters = %d Current_Loss = %.4f Mean Dice=%.4f Bladder Dice=%.4f Tumor Dice=%.4f Time = %.2f"\
                                    % (epoch, iters, loss.item(), mean_dice, bladder_dice, tumor_dice, time.time() - st)
                log.logger.info(string_print)
                st = time.time()
                writer.add_scalar('train_main_loss', loss.item(), iters)

            b_dice = b_dice / d_len
            t_dice = t_dice / d_len
            m_dice = (b_dice + t_dice) / 2

            log.logger.info('Epoch {}/{},Train Mean Dice {:.4}, Bladder Dice {:.4}, Tumor Dice {:.4}'.format(
                epoch, num_epoches, m_dice, b_dice, t_dice
            ))
            if (epoch % interval == 0) or (epoch == num_epoches):
                save_model(net, optimizer, epoch)
                if epoch == num_epoches:
                    writer.close() 
            if k_fold > 1:
                auto_val(net, val_loader, metric_fn) 
        except BaseException as e:
            log.logger.info(e)
            save_model(net, optimizer, num_epoches)
            writer.close() 
            log.logger.info('end training')
            return None
 

def auto_val(model, val_loader, metric_fn, is_show=False):
    model.eval()
    iters = 0
    SIZES = 8
    imgs = []
    preds = []
    gts = []
    dices = 0
    tumor_dices = 0
    bladder_dices = 0
    for i, (img, mask) in enumerate(val_loader):
        im = img
        img = img.to(device)
        model = model.to(device)
        pred = model(img)
        if LOSS:
            loss = criterion(pred, mask.to(device)).item()
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()
        iters += batch_size
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        bladder_dice = metric_fn(pred[:, 0:1, :], mask[:, 0:1, :], activation=None)
        tumor_dice = metric_fn(pred[:, 1:2, :], mask[:, 1:2, :], activation=None)
        mean_dice = (bladder_dice + tumor_dice) / 2
        dices += mean_dice
        tumor_dices += tumor_dice
        bladder_dices += bladder_dice
        acc = accuracy(pred, mask)
        p = precision(pred, mask)
        r = recall(pred, mask)
        log.logger.info('mean_dice={:.4}, bladder_dice={:.4}, tumor_dice={:.4}, acc={:.4}, p={:.4}, r={:.4}'.format(mean_dice.item(), bladder_dice.item(), tumor_dice.item(), acc, p, r))
        gt = mask.numpy()[0].transpose([1, 2, 0])
        gt = helpers.onehot_to_mask(gt, bladder.palette)
        pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
        pred = helpers.onehot_to_mask(pred, bladder.palette)
        im = im[0].numpy().transpose([1, 2, 0])
        if LOSS:
            writer.add_scalar('val_main_loss', loss, iters)

    val_mean_dice = dices / (len(val_loader) / batch_size)
    val_tumor_dice = tumor_dices / (len(val_loader) / batch_size)
    val_bladder_dice = bladder_dices / (len(val_loader) / batch_size)
    log.logger.info('Val Mean Dice = {:.4}, Val Bladder Dice = {:.4}, Val Tumor Dice = {:.4}'.format(val_mean_dice, val_bladder_dice, val_tumor_dice))

 
if __name__ == '__main__':
    main()
