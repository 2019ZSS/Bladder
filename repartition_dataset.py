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
 

def partition_data_v2(dataset_dir, output_root, is_need=False):
    """"is_need = True表示加入不干净的数据""" 
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
        if is_need:
            tmp_x_path = x_pre_path + '/' + str(i + 1) + '/unclear_bladder'
            tmp_y_path = y_pre_path + '/' + str(i + 1) + '/unclear_bladder'
            for path, _, filelist in os.walk(tmp_x_path):
                for filename in filelist:
                    # x_img = os.path.join(tmp_x_path, filename)
                    # y_img = os.path.join(tmp_y_path, filename)
                    # image_names.append((x_img, y_img))
                    image_names.append(str(i + 1) + '/unclear_bladder/' + filename)

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


def partition_data_v4(dataset_dir, output_root):
    dicom_dir = os.path.join(dataset_dir, 'Dicom')
    label_dir = os.path.join(dataset_dir, 'Label(Predicted)')

    dir_name = os.listdir(dicom_dir)
    dir_name = [x for x in dir_name if '(None)' not in x]
    random.shuffle(dir_name)

    val_size = 0.2
    train_names = []
    val_names = []

    rawdata_size = len(dir_name)
    random.seed(361)
    val_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * val_size))
    train_indices = [int(dir_name[i]) for i in range(rawdata_size) if i not in val_indices]
    val_indices = [int(dir_name[i]) for i in val_indices]
    

    for i in train_indices:
        for x in os.listdir(os.path.join(dicom_dir, str(i))):
            x = x.replace('.dcm', '')
            train_names.append(str(i) + '_' + x)
    
    for i in val_indices:
        for x in os.listdir(os.path.join(dicom_dir, str(i))):
            x = x.replace('.dcm', '')
            val_names.append(str(i) + '_' + x)
    
    with open(os.path.join(output_root, 'train.txt'), 'w') as f:
        for x in train_names:
            f.write(x + '\n')

    with open(os.path.join(output_root, 'val.txt'), 'w') as f:
        for x in val_names:
            f.write(x + '\n')      

    return train_names, val_names


def get_image_paths(index, image_dir, unclear=False):
    image_path = []
    pre_dir = os.path.join(image_dir, 'clear_bladder')
    for file in os.listdir(pre_dir):
        image_path.append('/'.join([str(index), 'clear_bladder', file]))
    if unclear:
        pre_dir  = os.path.join(image_dir, 'unclear_bladder')
        for file in os.listdir(pre_dir):
            image_path.append('/'.join([str(index), 'unclear_bladder', file]))
    return image_path


def get_train_val_data(dataset_dir, output_root, val_size=0.2, unclear=False):
    '''
    按照病人直接划分数据集和验证集，不作交叉验证
    '''
    dataset_dir = os.path.join(dataset_dir, 'Label(split_by_people&class)')
    dirs = [item for item in os.listdir(dataset_dir)]
    rawdata_size = len(dirs)
    val_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * val_size))
    train_indices = [i for i in range(0, rawdata_size) if i not in val_indices]
    
    with open(os.path.join(output_root, 'val.txt'), 'w') as f:
        for i in val_indices:
            image_dir = os.path.join(dataset_dir, dirs[i])
            print(image_dir)
            image_path = get_image_paths(dirs[i], image_dir, unclear)
            for item in image_path:
                f.write(item)
                f.write('\n')
    
    with open(os.path.join(output_root, 'train.txt'), 'w') as f:
        for i in train_indices:
            image_dir = os.path.join(dataset_dir, dirs[i])
            image_path = get_image_paths(dirs[i], image_dir, unclear)
            for item in image_path:
                f.write(item)
                f.write('\n')


def get_k_fold_data(dataset_dir, output_root, k=10, test_size=0.2, unclear=False):
    '''k-折交叉验证
    k: 几折
    val_size：验证集合所占的比例
    unclear: 不清晰的数据是否需要加入
    '''
    assert k > 1
    dataset_dir = os.path.join(dataset_dir, 'Label(split_by_people&class)')
    dirs = [item for item in os.listdir(dataset_dir)]
    rawdata_size = len(dirs)
    test_indices = random.sample(range(0, rawdata_size), math.floor(rawdata_size * test_size))
    train_indices = [i for i in range(0, rawdata_size) if i not in test_indices]

    output_dir = os.path.join(output_root, '{}_fold'.format(k))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for i in test_indices:
            image_dir = os.path.join(dataset_dir, dirs[i])
            image_path = get_image_paths(dirs[i], image_dir, unclear)
            for item in image_path:
                f.write(item)
                f.write('\n')

    # 每个训练集合的平均病人数量, 上取整
    train_avg = math.floor((rawdata_size - len(test_indices)) / k) 
    cnt, indices = 0, []

    def get_kth_train_dataset(indices):
        train_images = []
        for i in indices:
            image_dir = os.path.join(dataset_dir, dirs[i])
            train_images += get_image_paths(dirs[i], image_dir, unclear)
        return train_images
        
    k_train_images = []
    for item in train_indices:
        indices.append(item)
        if len(indices) == train_avg:
            k_train_images.append(get_kth_train_dataset(indices))
            indices = []
            cnt = cnt + 1
    
    # 把最后不均匀的一部分整体写入
    if len(indices) > 0:
        k_train_images.append(get_kth_train_dataset(indices))
    
    for i in range(len(k_train_images)):
        with open(os.path.join(output_dir, 'train_{}.txt'.format(i)), 'w') as f1:
            for j, train_images in enumerate(k_train_images):
                if i == j:
                    with open(os.path.join(output_dir, 'val_{}.txt'.format(i)), 'w') as f2:
                        for item in train_images:
                            f2.write(item)
                            f2.write('\n')
                else:
                    for item in train_images:
                        f1.write(item)
                        f1.write('\n')


if __name__ == '__main__':
    print('partition_data')

    dataset_dir = './hospital_data/2d'
    output_root = './hospital_data/2d'
    # get_train_val_data(dataset_dir, output_root)
    # get_k_fold_data(dataset_dir=dataset_dir, output_root=output_root, k=10)
    # train_names,  val_names = partition_data_v2(dataset_dir, output_root, is_need=False)
    # print(len(train_names))
    # # print(train_names)
    # print(len(val_names))
    # print(val_names)

    # dataset_dir = './hospital_data/3d/GENERAL'
    # output_root = './hospital_data/3d'
    # partition_data_v3(dataset_dir, output_root)   

    # dataset_dir = './hospital_data/MRI_T2'
    # output_root = './hospital_data/MRI_T2'
    # partition_data_v4(dataset_dir, output_root)
