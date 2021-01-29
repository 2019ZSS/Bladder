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
 
if __name__ == '__main__':
    dataset_dir = './data/'
    output_root = './data/'
    train_names,  val_names = partition_data(dataset_dir, output_root)
    print(len(train_names))
    print(train_names)
    print(len(val_names))
    print(val_names)