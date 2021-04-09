## 医学图像分割-膀胱壁分割

### 项目简介

> 目前是在二维数据集合上进行图像分割，之后要迁移学习到三维数据上进行图像分割, 有余力再考虑一下分期

### 项目环境

```python
conda create -n Bladder python=3.7 pytorch torchvision cudatoolkit=11.0 -c pytorch
(cuda根据自己的电脑版本微调9.0, 10.2)
conda activate Bladder
pip install -r requirements.txt
```

### 项目文件目录

```python
/                                        
├─ data                                  # 网络数据，公开上传
│  ├─ Images                             
│  ├─ Labels                             
│  ├─ train.txt                          
│  └─ val.txt                            
├─ hospital_data                         # 隐私数据, 不进行上传
│  ├─ 2d                                 # 2d数据
│  │  ├─ Image(split_by_people&class)    
│  │  └─ Label(split_by_people&class)    
│  ├─ MRI_T2                             # 待分割的3d数据
│  │  ├─ Dicom                           # 训练的dicom文件
│  │  └─ Label(Predicted)                # dicom对应的标签数据
│  ├─ 3d                                 # 3d数据
│  └─ pred                               # 保存预测的nrrd文件
├─ log                                   # 训练记录
│  └─ bladder_trainlog                   
│     └─ bladder_exp                     
├─ model                                 # 模型记录
│  └─ checkpoint                         
│     └─ exp                             
├─ utils                                 # 数据处理工具包
│  ├─ __init__.py                        
│  ├─ helpers.py                         # 主要是one-hot编码的实现和反解码
│  ├─ joint_transforms.py                # 2d图像预处理，例如图像裁剪，翻转, 以后可以考虑利用torch自带的功能进行实现
│  ├─ loss.py                            # 网络的损失函数
│  ├─ metrics.py                         # 网络的评价指标
│  ├─ tools.py                           # 网络训练中的一些参数输出, 以后可以改进成日志文件记录
│  └─ transforms.py                      # 图像处理, 从numpy转换成torch
├─ .gitignore                            # 项目忽略文件
├─ README.md                             # 项目说明
├─ bladder.py                            # 2d数据读入
├─ dicom_bladder.py                      # dcm数据读入
├─ dicom_data.py                         # 处理dicom数据一些函数
├─ dicom_test.py                         # 利用训练好的2d网络进行dcm数据的预测
├─ dicom_train.py                        # dicom上的训练
├─ dicom_validate.py                     # dicom的验证
├─ repartition_dataset.py                # 预处理数据集
├─ requirements.txt                      # 环境包依赖
├─ train.py                              # 数据训练
├─ u_net.py                              # u_net网络结构实现
└─ validate.py                           # 数据验证

```

### 项目运行

#### prepared_data

```python
# 调用repartition_dataset.py利用对应的数据生成函数，生成相对应的数据
python repartition_dataset.py
```

#### train(参考train.py里面的参数设置，最简单就是按照默认)

```python

# default
python train.py

# setting params
python train.py --data_path './hospital_data/2d' --epochs 100 --k_fold 10  --optimizer_name Adam --loss_name bcew_ --scale_size 256 -crop_size 256

# about params detail(python train.py --help you can can in command)
usage: train.py [-h] [--data_path DATA_PATH]
                [--pred_saved_path PRED_SAVED_PATH] [--epochs EPOCHS]
                [--interval INTERVAL] [--k_fold K_FOLD]
                [--batch_size BATCH_SIZE] [--n_cpu N_CPU] [--n_gpu N_GPU]
                [--scale_size SCALE_SIZE] [--crop_size CROP_SIZE]
                [--model_name MODEL_NAME] [--optimizer_name OPTIMIZER_NAME]
                [--metric METRIC] [--loss_name LOSS_NAME] [--LR_INIT LR_INIT]
                [--extra_description EXTRA_DESCRIPTION] [--resume RESUME]
                [--checkpoint_file CHECKPOINT_FILE] [--LOSS LOSS]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        load input data path
  --pred_saved_path PRED_SAVED_PATH
                        pred 2d-iamge saved path
  --epochs EPOCHS       number of epochs
  --interval INTERVAL   How long to save the model
  --k_fold K_FOLD       k fold training
  --batch_size BATCH_SIZE
                        size of each image batch
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --n_gpu N_GPU         number of cpu threads to use during batch generation
  --scale_size SCALE_SIZE
                        scale size of input iamge
  --crop_size CROP_SIZE
                        crop size of input image
  --model_name MODEL_NAME
                        model name
  --optimizer_name OPTIMIZER_NAME
                        optimizer name
  --metric METRIC       evaluation bladder and tumor loss
  --loss_name LOSS_NAME
                        loss_name
  --LR_INIT LR_INIT     init learing rate
  --extra_description EXTRA_DESCRIPTION
                        some extra information about the model
  --resume RESUME       Whether to continue the training model
  --checkpoint_file CHECKPOINT_FILE
                        checkpoint file to resume the model
  --LOSS LOSS           Whether to record validate loss
(wff) stu@uai-W560-G20:~/wff/Bladder$ python train.py --h
usage: train.py [-h] [--data_path DATA_PATH] [--epochs EPOCHS]
                [--interval INTERVAL] [--k_fold K_FOLD]
                [--batch_size BATCH_SIZE] [--n_cpu N_CPU] [--n_gpu N_GPU]
                [--scale_size SCALE_SIZE] [--crop_size CROP_SIZE]
                [--model_name MODEL_NAME] [--optimizer_name OPTIMIZER_NAME]
                [--metric METRIC] [--loss_name LOSS_NAME] [--LR_INIT LR_INIT]
                [--extra_description EXTRA_DESCRIPTION] [--resume RESUME]
                [--checkpoint_file CHECKPOINT_FILE] [--LOSS LOSS]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        load input data path
  --epochs EPOCHS       number of epochs
  --interval INTERVAL   How long to save the model
  --k_fold K_FOLD       k fold training
  --batch_size BATCH_SIZE
                        size of each image batch
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --n_gpu N_GPU         number of cpu threads to use during batch generation
  --scale_size SCALE_SIZE
                        scale size of input iamge
  --crop_size CROP_SIZE
                        crop size of input image
  --model_name MODEL_NAME
                        model name
  --optimizer_name OPTIMIZER_NAME
                        optimizer name
  --metric METRIC       evaluation bladder and tumor loss
  --loss_name LOSS_NAME
                        loss_name
  --LR_INIT LR_INIT     init learing rate
  --extra_description EXTRA_DESCRIPTION
                        some extra information about the model
  --resume RESUME       Whether to continue the training model
  --checkpoint_file CHECKPOINT_FILE
                        checkpoint file to resume the model
  --LOSS LOSS           Whether to record validate loss
```

#### test(参考test.py里面的参数设置，最简单就是按照默认, 但是要保证训练模型存在)

```python
# default
python train.py

# setting params
python validate.py --data_path './hospital_data/2d' --epochs 100 --k_fold 10  --optimizer_name Adam --loss_name bcew_ --scale_size 256 -crop_size 256

# # about params detail(python validate.py --help you can can in command)
usage: validate.py [-h] [--data_path DATA_PATH] [--model_path MODEL_PATH]
                   [--pred_saved_path PRED_SAVED_PATH] [--epochs EPOCHS]
                   [--k_fold K_FOLD] [--batch_size BATCH_SIZE] [--n_cpu N_CPU]
                   [--n_gpu N_GPU] [--scale_size SCALE_SIZE]
                   [--crop_size CROP_SIZE] [--model_name MODEL_NAME]
                   [--optimizer_name OPTIMIZER_NAME] [--metric METRIC]
                   [--loss_name LOSS_NAME]
                   [--extra_description EXTRA_DESCRIPTION] [--LOSS LOSS]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        load input data path
  --model_path MODEL_PATH
                        model path to validate/test
  --pred_saved_path PRED_SAVED_PATH
                        pred 2d-iamge saved path
  --epochs EPOCHS       number of epochs
  --k_fold K_FOLD       k fold training
  --batch_size BATCH_SIZE
                        size of each image batch
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --n_gpu N_GPU         number of cpu threads to use during batch generation
  --scale_size SCALE_SIZE
                        scale size of input iamge
  --crop_size CROP_SIZE
                        crop size of input image
  --model_name MODEL_NAME
                        model name
  --optimizer_name OPTIMIZER_NAME
                        optimizer name
  --metric METRIC       evaluation bladder and tumor loss
  --loss_name LOSS_NAME
                        loss_name
  --extra_description EXTRA_DESCRIPTION
                        some extra information about the model
  --LOSS LOSS           Whether to record validate loss
```

#### dicom_test (dicom数据转化为nrrd文件，最终需要的结果)

```python
python dicom_test.py

usage: dicom_test.py [-h] [--data_path DATA_PATH] [--model_path MODEL_PATH]
                     [--pred_save_path PRED_SAVE_PATH] [--dicom_dir DICOM_DIR]
                     [--nrrd_out_dir NRRD_OUT_DIR] [--epochs EPOCHS]
                     [--k_fold K_FOLD] [--batch_size BATCH_SIZE]
                     [--n_cpu N_CPU] [--n_gpu N_GPU] [--scale_size SCALE_SIZE]
                     [--crop_size CROP_SIZE] [--model_name MODEL_NAME]
                     [--optimizer_name OPTIMIZER_NAME] [--metric METRIC]
                     [--loss_name LOSS_NAME]
                     [--extra_description EXTRA_DESCRIPTION] [--LOSS LOSS]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        load input data path
  --model_path MODEL_PATH
                        model path to validate/test
  --pred_save_path PRED_SAVE_PATH
                        saved path about the pred-2d-image
  --dicom_dir DICOM_DIR
                        dicom data path
  --nrrd_out_dir NRRD_OUT_DIR
                        nrrd file save path
  --epochs EPOCHS       number of epochs
  --k_fold K_FOLD       k fold training
  --batch_size BATCH_SIZE
                        size of each image batch
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --n_gpu N_GPU         number of cpu threads to use during batch generation
  --scale_size SCALE_SIZE
                        scale size of input iamge
  --crop_size CROP_SIZE
                        crop size of input image
  --model_name MODEL_NAME
                        model name
  --optimizer_name OPTIMIZER_NAME
                        optimizer name
  --metric METRIC       evaluation bladder and tumor loss
  --loss_name LOSS_NAME
                        loss_name
  --extra_description EXTRA_DESCRIPTION
                        some extra information about the model
  --LOSS LOSS           Whether to record validate loss
```