
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
from dicom_data import is_dicom_file, dcm_to_img_numpy
import train
import dicom_bladder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS = False 
np.set_printoptions(threshold=9999999)
batch_size = 1
data_path = './hospital_data/3d'
mode = 'test'

palette = [[128], [255], [0]]
joint_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(160),
    ])
val_input_transform = extended_transforms.ImgToTensor()
make_dataset_fn = dicom_bladder.make_dataset_dcm
val_set = dicom_bladder.Bladder(data_path, mode, make_dataset_fn=make_dataset_fn, 
                            joint_transform=joint_transform, transform=val_input_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


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


def test_val(model, img_path):
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
    # np.uint8()反归一化到[0, 255]
    imgs = np.uint8(np.hstack([mri, pred]))
    cv2.imshow("mri pred", imgs)
    cv2.waitKey(0)


if __name__ == "__main__":
    print('test')
    root = './hospital_data/3d'
    mode = 'test'
    imgs = make_dataset_fn(root=root, mode=mode)
    for img_path in imgs:
        test_val(model, img_path)
    


