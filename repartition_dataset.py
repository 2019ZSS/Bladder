# repartition_dataset.py
 
import os
import math
import random
 
def partition_data(dataset_dir, ouput_root):
    """
    Divide the raw data into training sets and validation sets
    :param dataset_dir: path root of dataset
    :param ouput_root: the root path to the output file
    :return:
    """
    image_names = []
    mask_names = []
    val_size = 0.2
    train_names = []
    val_names = []
 
    for file in os.listdir(os.path.join(dataset_dir, "Images")):
        image_names.append(file)
    image_names.sort()
    for file in os.listdir(os.path.join(dataset_dir, "Labels")):
        mask_names.append(file)
    mask_names.sort()
 
    rawdata_size = len(image_names)
    random.seed(361)
    val_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * val_size))
    train_indices = []
    for i in range(0, rawdata_size):
        if i not in val_indices:
            train_indices.append(i)
 
    with open(os.path.join(ouput_root, 'val.txt'), 'w') as f:
        for i in val_indices:
            val_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')
 
    with open(os.path.join(ouput_root, 'train.txt'), 'w') as f:
        for i in train_indices:
            train_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')
    train_names.sort(), val_names.sort()
    return train_names, val_names
 

def partition_data_v2(dataset_dir, output_root): 
    x_pre_path = dataset_dir + '/Image(split_by_people&class)'
    y_pre_path = dataset_dir + '/Label(split_by_people&class)'
    n = len(os.listdir(x_pre_path))

    image_names = []
    for i in range(n):
        tmp_x_path = x_pre_path + '/' + str(i + 1) + '/clear_bladder'
        tmp_y_path = y_pre_path + '/' + str(i + 1) + '/clear_bladder'
        for path, _, filelist in os.walk(tmp_x_path):
            for filename in filelist:
                # x_img = os.path.join(tmp_x_path, filename)
                # y_img = os.path.join(tmp_y_path, filename)
                # image_names.append((x_img, y_img))
                image_names.append(str(i + 1) + '/clear_bladder/' + filename)

    # 随机打乱
    random.shuffle(image_names)
    val_size = 0.2
    train_names = []
    val_names = []

    rawdata_size = len(image_names)
    random.seed(361)
    val_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * val_size))
    train_indices = []
    for i in range(0, rawdata_size):
        if i not in val_indices:
            train_indices.append(i)
    
    with open(os.path.join(output_root, 'val.txt'), 'w') as f:
        for i in val_indices:
            val_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')

    with open(os.path.join(output_root, 'train.txt'), 'w') as f:
        for i in train_indices:
            train_names.append(image_names[i])
            f.write(image_names[i])
            f.write('\n')
    train_names.sort(), val_names.sort()
    return train_names, val_names


def partition_data_v3(dataset_dir, output_root):
    dcm_dir = dataset_dir
    dcm_list = []
    for dirName, subdirList, fileList in os.walk(dataset_dir):
        for filename in fileList:
            #判断文件是否为.dcm文件
            if ".dcm" in filename.lower():  
                # 加入到列表中
                dcm_list.append(os.path.join(dirName,filename))

    # 随机打乱
    random.shuffle(dcm_list)
    test_size = 0.01
    test_names = []
    rawdata_size = len(dcm_list)
    random.seed(361)
    test_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * test_size))
    with open(os.path.join(output_root, 'test.txt'), 'w') as f:
        for i in test_indices:
            test_names.append(dcm_list[i])
            f.write(dcm_list[i])
            f.write('\n')

    return test_names


if __name__ == '__main__':
    print('partition_data')

    # dataset_dir = './data'
    # output_root = './data'
    # train_names,  val_names = partition_data(dataset_dir, output_root)
    # print(len(train_names))
    # # print(train_names)
    # print(len(val_names))
    # # print(val_names)

    # dataset_dir = './hospital_data/2d'
    # output_root = './hospital_data/2d'
    # train_names,  val_names = partition_data_v2(dataset_dir, output_root)
    # print(len(train_names))
    # # print(train_names)
    # print(len(val_names))
    # # print(val_names)

    dataset_dir = './hospital_data/3d/GENERAL'
    output_root = './hospital_data/3d'
    partition_data_v3(dataset_dir, output_root)