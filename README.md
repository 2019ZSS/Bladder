## 医学图像分割-膀胱壁分割

### 项目简介

> 目前是在二维数据集合上进行图像分割，之后要迁移学习到三维数据上进行图像分割

### 项目环境

```python
python3 + pytorch + cuda
```

### 项目文件目录

```python
--data
--utils(工具处理) 
--repartition_dataset.py(预处理数据集)
--bladder.py(数据集读入)
--u_net.py(网络结构)
--train.py(训练)
--validate.py(验证)
--requirements.txt(环境包依赖)
```

