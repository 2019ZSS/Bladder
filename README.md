## 医学图像分割-膀胱壁分割

### 项目简介

> 目前是在二维数据集合上进行图像分割，之后要迁移学习到三维数据上进行图像分割, 有余力再考虑一下分割

### 项目环境

```python
python=3.7.2, pytorch, cuda (装有就行, 版本很随意)
```

### 项目文件目录

```python
--data(网络数据，公开上传)
--hospital_data(隐私数据, 不进行上传)
---2d(2d数据) 
----Image(split_by_people&class)
----Label(split_by_people&class)
---3d(3d数据)
--utils(数据处理工具包)
---helpers.py(主要是one-hot编码的实现和反解码)
---joint_transforms.py(2d图像读入前的一些预处理，例如图像裁剪，翻转, 以后可以考虑利用torch自带的功能进行实现)
---loss.py(网络的损失函数)
---metrics.py(网络的评价指标)
---tools.py(网络训练中的一些参数输出, 以后可以改进成日志文件记录)
---transforms.py(图像处理, 从numpy转换成torch)
--.gitgnore (项目忽略文件) 
--bladder.py(2d数据读入)
--dicom_bladder.py(dcm数据读入)
--dicom_data.py(处理dicom数据一些函数)
--dicom_tset.py(利用训练好的2d网络进行dcm数据的预测)
--repartition_dataset.py(预处理数据集)
--train.py(数据训练)
--u_net.py(u_net网络结构实现)
--validate.py(数据验证)
--requirements.txt(环境包依赖, 可能会多出一部分用不到的包依赖，实际上只需要核心的pytorch, opencv, ptdicom有按照即可, 版本请随意, 不做严格要求)
```

