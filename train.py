# train.py
 
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'.'))

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
 
import bladder
from utils.loss import SoftDiceLoss, SoftDiceLossV2
from utils.metrics import diceCoeff, diceCoeffv2
from utils import tools
from u_net import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crop_size = 128
batch_size = 2
n_epoch = 30
model_name = 'U_Net_'
loss_name = 'dice_'
times = 'no_' + str(n_epoch)
extra_description = ''
writer = SummaryWriter(os.path.join('./log/bladder_trainlog',  'bladder_exp', model_name+loss_name+times+extra_description))
resume = False

def main():
    net = U_Net(img_ch=1, num_classes=3).to(device)
    
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(256),
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip()
    ])
    center_crop = joint_transforms.CenterCrop(crop_size)
    train_input_transform = extended_transforms.ImgToTensor()
 
    target_transform = extended_transforms.MaskToTensor()
    train_set = bladder.Bladder('./data', 'train',
                                joint_transform=train_joint_transform, center_crop=center_crop,
                                transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
 
    if loss_name == 'dice_':
        criterion = SoftDiceLossV2(activation='sigmoid', num_classes=3).to(device)
    elif loss_name == 'bce_':
        criterion = nn.BCELoss().to(device)
    elif loss_name == 'wbce_':
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif loss_name == 'ce_':
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    train(train_loader, net, criterion, optimizer, n_epoch, 0)
 
 
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
    main()