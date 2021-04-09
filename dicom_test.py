
import os 
import cv2
from PIL import Image
import utils.joint_transforms as joint_transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from utils import helpers
import utils.transforms as extended_transforms
from utils.metrics import *
from utils.loss import *
from u_net import U_Net
from dicom_data import (
    is_dicom_file, 
    dcm_to_img_numpy, 
    get_dcm_list,
)
import dicom_bladder
import numpy as np 
import nrrd
from utils import tools
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./hospital_data/3d', help='load input data path')
parser.add_argument("--model_path", type=str, default='', help='model path to validate/test')
parser.add_argument('--pred_save_path', type=str, default='./hospital_data/3d_pred', help='saved path about the pred-2d-image')
parser.add_argument('--dicom_dir', type=str, default='./hospital_data/MRI_T2/Dicom', help='dicom data path')
parser.add_argument('--nrrd_out_dir', type=str, default='./hospital_data/pred', help='nrrd file save path')
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
pred_save_path = opt.pred_save_path
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
if not os.path.exists(model_saved_dir):
    os.makedirs(model_saved_dir)

if not model_path:
    model_path = os.path.join(model_saved_dir, '{}.pth'.format(n_epoch))

model = U_Net(img_ch=1, num_classes=3).to(device)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model.cuda())
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

np.set_printoptions(threshold=9999999)
mode = 'test'
palette = [[128], [255], [0]]
joint_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scale_size),
        # transforms.CenterCrop(crop_size),
])
val_input_transform = extended_transforms.ImgToTensor()
# make_dataset_fn = dicom_bladder.make_dataset_dcm
# val_set = dicom_bladder.Bladder(data_path, mode, make_dataset_fn=make_dataset_fn, 
#                             joint_transform=joint_transform, transform=val_input_transform)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

if LOSS:
    writer = SummaryWriter(os.path.join('./log/testexp', 'bladder_exp', model_name+loss_name+times+extra_description))

if loss_name == 'dice_':
    criterion = SoftDiceLossV2(activation='sigmoid', num_classes=3).to(device)
elif loss_name == 'bcew_':
    criterion = nn.BCEWithLogitsLoss().to(device)
else:
    raise NotImplementedError('{} NotImplemented '.format(loss_name))


def test_val(model, img_path, is_show=False):
    if not is_dicom_file(img_path):
        img = Image.open(img_path)
        img = np.array(img)
    else:
        img = dcm_to_img_numpy(img_path)

    img = joint_transform(img)
    img = np.array(img)
    img = np.expand_dims(img, axis=2)
    mri = img
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
    pred = pred.cpu().detach().numpy()[0].transpose([1, 2, 0])
    # 用来看预测的像素值
    pred_showval = pred
    pred = helpers.onehot_to_mask(pred, dicom_bladder.palette)
    if is_show:
        # np.uint8()反归一化到[0, 255]
        imgs = np.uint8(np.hstack([mri, pred]))
        cv2.imshow("mri pred", imgs)
        cv2.waitKey(0)
    # pred.shape (H, W, 1)
    return mri, pred


def merge_mri_pred(mri, pred):
    """mri原图和pred图像融合"""
    idx = (pred == 255)
    mri[idx] = pred[idx]
    return mri 


def dcm_pred(dicom_path, out_path):
    """dicom文件预测，并以nrdd格式进行保存
    dicom_path: dicom文件保存路径
    out_path: 输出路径
    """
    try:
        dcm_list = get_dcm_list(dicom_path)
    except Exception as e:
        print(e)
        return 

    data = []
    pred_data = []
    bladder_data = []
    tumor_data = []
    for img_path in dcm_list:
        mri, pred = test_val(model, img_path, is_show=False)
        mri = mri.transpose([2, 0, 1])
        pred = pred.transpose([2, 0, 1])
        # data.append(merge_mri_pred(mri, pred))
        data.append(mri)
        pred_data.append(pred)
        # print(pred.shape)
        tmp = np.zeros(pred.shape)
        tmp[pred == 128] = 128
        bladder_data.append(tmp)

        tmp = np.zeros(pred.shape)
        tmp[pred == 255] = 255
        tumor_data.append(tmp)

        # imgs = np.uint8(np.hstack([bladder_data[-1].transpose([1, 2, 0]), tumor_data[-1].transpose([1, 2, 0])]))
        # cv2.imshow("bladder tumor", imgs)
        # cv2.waitKey(0)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    data = np.concatenate(data)
    pred_data = np.concatenate(pred_data)
    bladder_data = np.concatenate(bladder_data)
    tumor_data = np.concatenate(tumor_data)
    nrrd.write(os.path.join(out_path, 'origin.nrrd'), data, index_order='C')
    nrrd.write(os.path.join(out_path, 'pred.nrrd'), pred_data, index_order='C')
    nrrd.write(os.path.join(out_path, 'bladder.nrrd'), bladder_data, index_order='C')
    nrrd.write(os.path.join(out_path, 'tumor.nrrd'), tumor_data, index_order='C')


def show_pred_nrdd(origin_path, bladder_path, tumor_path): 
    '''查看预测的nrrd文件在每一帧上的效果'''
    origin_data, _ = nrrd.read(origin_path)
    bladder_data, _ = nrrd.read(bladder_path)
    tumor_data, _ = nrrd.read(tumor_path)

    assert origin_data.shape == bladder_data.shape
    assert origin_data.shape == tumor_data.shape
    size = origin_data.shape[2]

    def get_2d_data(data):
        img = data[:,:,i]
        img = img.swapaxes(0, 1)
        img = np.expand_dims(img, axis=2)
        return img 

    for i in range(size):
        origin = get_2d_data(origin_data)
        bla = get_2d_data(bladder_data)
        tumor = get_2d_data(tumor_data)
        imgs = np.uint8(np.hstack([origin, bla, tumor]))
        cv2.imshow("origin bladder tumor", imgs)
        cv2.waitKey(0)


if __name__ == "__main__":
    print('test')
    # root = './hospital_data/3d'
    # mode = 'test'
    # imgs = make_dataset_fn(root=root, mode=mode)
    # for img_path in imgs:
    #     test_val(model, img_path, is_show=True)

    
    # dicom_path = './hospital_data/MRI_T2/Dicom/1'
    # out_path = './hospital_data/pred'
    # dcm_pred(dicom_path, out_path)

    dicom_dir = opt.dicom_dir
    out_dir = opt.nrrd_out_dir
    dicom_names = os.listdir(dicom_dir)
    for dicom_name in dicom_names:
        dicom_path = os.path.join(dicom_dir, dicom_name)
        out_path = os.path.join(out_dir, dicom_name)
        dcm_pred(dicom_path, out_path)
