# validate.py
import os  
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
from u_net import *
from utils import tools
import re 
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='./hospital_data/2d', help='load input data path')
parser.add_argument("--model_path", type=str, default='', help='model path to validate/test')
parser.add_argument("--pred_saved_path", type=str, default='', help="pred 2d-iamge saved path")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--k_fold", type=int, default=1, help='k fold training')
parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0,1', help="number of cpu threads to use during batch generation")
parser.add_argument("--scale_size", type=int, default=256, help='scale size of input iamge')
parser.add_argument("--crop_size", type=int, default=256, help='crop size of input image')
parser.add_argument("--model_name", type=str, default='U_Net', help='model name')
parser.add_argument("--optimizer_name", type=str, default='Adam', help='optimizer name')
parser.add_argument("--metric", type=str, default='dice_coef', help='evaluation bladder and tumor loss')
parser.add_argument("--loss_name", type=str, default='bcew_', help='loss_name')
parser.add_argument("--extra_description", type=str, default='', help='some extra information about the model')
parser.add_argument("--LOSS", type=bool, default=False, help='Whether to record validate loss')
opt = parser.parse_args()
num_workers = opt.n_cpu
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
data_path = opt.data_path
k_fold = opt.k_fold
model_path = opt.model_path
pred_saved_path = opt.pred_saved_path
scale_size = opt.scale_size
crop_size = opt.crop_size
batch_size = opt.batch_size
n_epoch = opt.epochs
model_name = opt.model_name
optimizer_name = opt.optimizer_name
metric = opt.metric
loss_name = opt.loss_name
extra_description = opt.extra_description
LOSS = opt.LOSS
times = 'no_' + str(n_epoch)

def generate_dir(prefix,model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description):
    keys = [model_name, optimizer_name, loss_name, metric, '{}_fold'.format(k_fold), str(scale_size), extra_description]
    return os.path.join(prefix, '_'.join(keys))

model_saved_dir = generate_dir('./model/checkpoint/exp', model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description)

log_saved_dir = generate_dir('./log/eval', model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description)
if not os.path.exists(log_saved_dir):
    os.makedirs(log_saved_dir)

log_file = log_saved_dir + times + '.log'
log = tools.Logger(filename=log_file)

if not pred_saved_path:
    pred_saved_path = generate_dir('./hospital_data/eval', model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description)

if not os.path.exists(pred_saved_path):
    os.makedirs(pred_saved_path)

if not model_path:
    model_path = os.path.join(model_saved_dir, '{}.pth'.format(n_epoch))

model = U_Net(img_ch=1, num_classes=3).to(device)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model.cuda())
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# numpy 高维数组打印不显示...
np.set_printoptions(threshold=9999999)
palette = [[128], [255], [0]]
val_input_transform = extended_transforms.ImgToTensor()
center_crop = joint_transforms.Compose([
    joint_transforms.Scale(scale_size),
    joint_transforms.CenterCrop(crop_size)
])
 
target_transform = extended_transforms.MaskToTensor()
make_dataset_fn = bladder.make_dataset_v2
if k_fold > 1:
    mode = 'test'
    mode_root = os.path.join(data_path, '{}_fold'.format(k_fold))
else:
    mode = 'val'
    mode_root = None
val_set = bladder.Bladder(root=data_path, mode=mode, mode_root=mode_root,
                            transform=val_input_transform, center_crop=center_crop,
                            target_transform=target_transform, make_dataset_fn=make_dataset_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)


if LOSS:
    writer = SummaryWriter(generate_dir('./log/eval/bladder_exp', model_name, optimizer_name, loss_name, metric, k_fold, scale_size, extra_description))

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


def eval(cnt, model, img_path, mask_path):
    img = Image.open(img_path)
    mask = Image.open(mask_path)
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
    log.logger.info('mean_dice={:.4}, bladder_dice={:.4}, tumor_dice={:.4}, acc={:.4}, p={:.4}, r={:.4}'.format(mean_dice.item(), bladder_dice.item(), tumor_dice.item(), acc.item(), p.item(), r.item()))
    pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
    if val_input_transform.name == 'ImgToTensorV2':
        pred = np.uint8(pred * 255)
    # 用来看预测的像素值
    pred_showval = pred
    pred = helpers.onehot_to_mask(pred, bladder.palette)
    # np.uint8()反归一化到[0, 255]
    imgs = np.uint8(np.hstack([mri, pred, mask]))
    if not os.path.exists(pred_saved_path):
        os.makedirs(pred_saved_path)
    tmp = re.search(r'\d+\s{1,}\(\d+\).png', img_path)
    if hasattr(tmp, 'group'):
        suffix = tmp.group()
    else:
        suffix = str(cnt) + '.png'
    cv2.imwrite(os.path.join(pred_saved_path, suffix), imgs, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]) 


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
    root = data_path
    imgs = make_dataset_fn(root=root, mode=mode, mode_root=mode_root)
    for i, (img_path, mask_path) in enumerate(imgs):
        eval(i, model, img_path, mask_path)
    auto_val(model, val_loader, metric_fn)