# validate.py
  
import cv2
from PIL import Image
import utils.joint_transforms as joint_transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import bladder
from utils import helpers
import utils.transforms as extended_transforms
from utils.metrics import *
from utils.loss import *
from u_net import U_Net
import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS = False 
# numpy 高维数组打印不显示...
np.set_printoptions(threshold=9999999)
batch_size = 1

palette = [[128], [255], [0]]
val_input_transform = extended_transforms.ImgToTensor()
center_crop = joint_transforms.Compose([
    joint_transforms.Scale(256),
    joint_transforms.CenterCrop(128)]
)
 
target_transform = extended_transforms.MaskToTensor()
val_set = bladder.Bladder('./data', 'val',
                              transform=val_input_transform, center_crop=center_crop,
                              target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
 
# 验证用的模型名称
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
elif loss_name == 'bce_':
    criterion = nn.BCELoss().to(device)
elif loss_name == 'wbce_':
    criterion = nn.BCEWithLogitsLoss().to(device)
elif loss_name == 'ce_':
    criterion = nn.CrossEntropyLoss().to(device)


def val(model):
    imname = '2-IM131'
    img = Image.open('./data/Images/{}.png'.format(imname))
    mask = Image.open('./data/Labels/{}.png'.format(imname))
    img, mask = center_crop(img, mask)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=2)
    mri = img
    mask = np.asarray(mask)
    mask = np.expand_dims(mask, axis=2)
 
    gt = np.float32(helpers.mask_to_onehot(mask, bladder.palette))
    # 用来看gt的像素值
    gt_showval = gt
    gt = np.expand_dims(gt, axis=3)
    gt = gt.transpose([3, 2, 0, 1])
    gt = torch.from_numpy(gt)

    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, axis=3)
    img = img.transpose([3, 0, 1, 2])
    img = val_input_transform(img)

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
    # 用来看预测的像素值
    pred_showval = pred
    pred = helpers.reverse_one_hot(pred)
    pred = helpers.colour_code_segmentation(pred, palette)
    # np.uint8()反归一化到[0, 255]
    imgs = np.uint8(np.hstack([pred, mask]))
    cv2.imshow("mri pred gt", imgs)
    cv2.waitKey(0)
 

def auto_val(model):
    # 效果展示图片数
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
        bladder_dice = diceCoeff(pred[:, 0:1, :], mask[:, 0:1, :], activation=None)
        tumor_dice = diceCoeff(pred[:, 1:2, :], mask[:, 1:2, :], activation=None)
        mean_dice = (bladder_dice + tumor_dice) / 2
        dices += mean_dice
        tumor_dices += tumor_dice
        bladder_dices += bladder_dice
        acc = accuracy(pred, mask)
        p = precision(pred, mask)
        r = recall(pred, mask)
        print('mean_dice={:.4}, bladder_dice={:.4}, tumor_dice={:.4}, acc={:.4}, p={:.4}, r={:.4}'
              .format(mean_dice.item(), bladder_dice.item(), tumor_dice.item(),
                      acc, p, r))
        gt = mask.numpy()[0].transpose([1, 2, 0])
        gt = helpers.onehot_to_mask(gt, bladder.palette)
        pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
        pred = helpers.onehot_to_mask(pred, bladder.palette)
        im = im[0].numpy().transpose([1, 2, 0])
        if LOSS:
            writer.add_scalar('val_main_loss', loss, iters)
        if len(imgs) < SIZES:
            imgs.append(im * 255)
            preds.append(pred)
            gts.append(gt)
    val_mean_dice = dices / (len(val_loader) / batch_size)
    val_tumor_dice = tumor_dices / (len(val_loader) / batch_size)
    val_bladder_dice = bladder_dices / (len(val_loader) / batch_size)
    print('Val Mean Dice = {:.4}, Val Bladder Dice = {:.4}, Val Tumor Dice = {:.4}'
          .format(val_mean_dice, val_bladder_dice, val_tumor_dice))

    imgs = np.hstack([*imgs])
    preds = np.hstack([*preds])
    gts = np.hstack([*gts])
    show_res = np.vstack(np.uint8([imgs, preds, gts]))
    cv2.imshow("top is mri , middle is pred,  bottom is gt", show_res)
    cv2.waitKey(0)


if __name__ == '__main__':
    # val(model)
    auto_val(model)