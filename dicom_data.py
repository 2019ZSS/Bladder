# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pydicom
import numpy as np 
import cv2
import SimpleITK
import png
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
from PIL import Image, ImageOps


def get_dcm_list(PathDicom):
    """
    PathDicom: dicom 文件路径
    """
    # 用lstFilesDCM作为存放DICOM files的列表
    lstFilesDCM = []

    for dirName,subdirList,fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  #判断文件是否为dicom文件
                lstFilesDCM.append(os.path.join(dirName,filename)) # 加入到列表中

    if len(lstFilesDCM) == 0:
        raise("{}目录下没有dcm文件, 请检查目录是否正确".format(PathDicom))

    return lstFilesDCM


def show_3d_dicom(PathDicom):
    """.dcm文件内容
    PathDicom: dicom 文件路径
    """
    lstFilesDCM = get_dcm_list(PathDicom)
    
    ## 将第一张图片作为参考图
    RefDs = pydicom.read_file(lstFilesDCM[0])   #读取第一张dicom图片

    # 建立三维数组
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    
    # 得到spacing值 (mm为单位)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    
    # 三维数据
    # 0到（第一个维数加一*像素间的间隔），步长为constpixelSpacing
    x = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0]) 
    y = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1]) 
    z = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2]) 
    
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    
    # 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM)
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    

    # 轴状面显示
    for i in range(ArrayDicom.shape[2]):
        # print(ArrayDicom[:, :, i])
        plt.figure(dpi=300)
        plt.axes().set_aspect('equal', 'datalim')
        plt.set_cmap(plt.gray())
        # 第三个维度表示现在展示的是第几层
        plt.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, i])) 
        plt.show()

    # 冠状面显示
    # plt.figure(dpi=300)
    # plt.axes().set_aspect('equal', 'datalim')
    # plt.set_cmap(plt.gray())
    # plt.pcolormesh(z, x, np.flipud(ArrayDicom[:, 150, :]))
    # plt.show()


def show_dicom_dir(dicom_dir_path):
    """查看dicomdir文件内容
    dicom_dir_path: dicomdir 
    """
    dicom_dir = pydicom.filereader.read_dicomdir(dicom_dir_path)
    base_dir = os.path.dirname(dicom_dir_path)

    #go through the patient record and print information
    for patient_record in dicom_dir.patient_records:
        if hasattr(patient_record, 'PatientID') and hasattr(patient_record, 'PatientName'):
            print("Patient: {}: {}".format(patient_record.PatientID, patient_record.PatientName))

        studies = patient_record.children
        # got through each serie
        for study in studies:
            print(" " * 4 + "Study {}: {}: {}".format(study.StudyID, study.StudyDate, study.StudyDescription))
            all_series = study.children
            # go through each serie
            for series in all_series:
                image_count = len(series.children)
                plural = ('', 's')[image_count > 1]

                # Write basic series info and image count

                # Put N/A in if no Series Description
                if 'SeriesDescription' not in series:
                    series.SeriesDescription = "N/A"

                print(" " * 8 + "Series {}: {}: {} ({} image{})".format(
                    series.SeriesNumber, series.Modality, series.SeriesDescription,
                    image_count, plural))

                # Open and read something from each image, for demonstration
                # purposes. For simple quick overview of DICOMDIR, leave the
                # following out
                print(" " * 12 + "Reading images...")
                image_records = series.children
                image_filenames = [os.path.join(base_dir, *image_rec.ReferencedFileID) for image_rec in image_records]

                datasets = [pydicom.dcmread(image_filename) for image_filename in image_filenames]

                patient_names = set(ds.PatientName for ds in datasets)
                patient_IDs = set(ds.PatientID for ds in datasets)

                # List the image filenames
                print("\n" + " " * 12 + "Image filenames:")
                # print(" " * 12, end=' ')
                # pprint(image_filenames, indent=12)

                # Expect all images to have same patient name, id
                # Show the set of all names, IDs found (should each have one)
                print(" " * 12 + "Patient Names in images..: {}".format(patient_names))
                print(" " * 12 + "Patient IDs in images..: {}".format(patient_IDs))



def show_all_dicom_dir():
    base_dir = r'.\hospital_data\3d\GENERAL'
    dicom_dir_list = []
    for dirName,subdirList,fileList in os.walk(base_dir):
        for filename in fileList:
            #判断文件是否为DICOMDIR文件
            if "DICOMDIR" in filename.upper():  
                # 加入到列表中
                dicom_dir_list.append(os.path.join(dirName,filename)) 
    for dicom_dir_path in dicom_dir_list:
        show_dicom_dir(dicom_dir_path)



def is_dicom_file(filename):
    """判断某文件是否是dicom格式的文件"""
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False


def load_patient(src_dir):
    '''
    读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            item = pydicom.read_file(src_dir + '/' + s)
            slices.append(item)

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
    读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array


def dicm_to_png(dicom_dir, png_dir):
    """
    dicom文件转成png
    dicom_dir: dicom文件目录
    png_dir: png文件目录
    """
    # 读取dicom文件的元数据(dicom tags)
    slices = load_patient(dicom_dir)
    print('The number of dicom files : ', len(slices))
    # 提取dicom文件中的像素值
    image = get_pixels_hu_by_simpleitk(dicom_dir)
    for i in tqdm(range(image.shape[0])):
    	#输出png文件目录
        img_path = png_dir + '\\' + str(i).rjust(4, '0') + "_i.png"
        org_img = image[i] * 20
        # 保存图像数组
        cv2.imwrite(img_path, org_img)

    print('finished')


def dcm_to_img_numpy(in_path):
    """把输入的dcm文件转换成对应的numpy
    """
    ds = pydicom.dcmread(in_path)

    if 'WindowWidth' in ds:
        imgae_array = ds.pixel_array
    
    imgae_array = pydicom.pixel_data_handlers.util.apply_voi_lut(ds.pixel_array, ds)

    shape = imgae_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = imgae_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    return image_2d_scaled


def dcm_convert_to_png(in_path, out_path):
    """
    in_path: 转换输入的dcm文件路径
    out_path: 输出保存的png文件路径
    """
    ds = pydicom.dcmread(in_path)

    if 'WindowWidth' in ds:
        imgae_array = ds.pixel_array
    
    imgae_array = pydicom.pixel_data_handlers.util.apply_voi_lut(ds.pixel_array, ds)

    shape = imgae_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = imgae_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file
    with open(out_path, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)


def dicom_to_png(dicom_dir):
    # 用lstFilesDCM作为存放DICOM files的列表
    dicom_list = []

    for dir_name, sub_dir_list, file_list in os.walk(dicom_dir):
        for filename in file_list:
            if ".dcm" in filename.lower(): 
                dicom_list.append(os.path.join(dir_name, filename))
                

    if len(dicom_list) == 0:
        raise("{}目录下没有dcm文件, 请检查目录是否正确".format(dicom_dir))


    base_path = r'.\hospital_data\dicom_to_png'
    for i, dcm_path in enumerate(dicom_list):
        out_path = base_path + '\\' + str(i) + '.png'
        dcm_convert_to_png(dcm_path, out_path)


def show_png(png_path):
    img = Image.open(png_path)

    
    print(img.format)
    print(img.size)
    # w, h = 512, 512
    # img = img.resize((w, h), Image.BILINEAR)
    # mat = np.array(img)
    # print(mat)
    # row, col = mat.shape
    # for i in range(row):
    #     for j in range(col):
    #         if mat[i, j] > 0:
    #             print(i, j, mat[i, j])
    plt.figure('black-white')
    plt.imshow(img)
    plt.show()
    # img.save('.\output.png')


if __name__ == "__main__":
    print('3d data')
    # PathDicom = "./hospital_data/3d/GENERAL/1.2.528.1.1001.200.2033.2157.206915881851013.20190521085533187/SDY00000/SRS00000" 
    # dicom_dir_path = r'.\hospital_data\3d\GENERAL\1.2.528.1.1001.200.2033.2157.206915881851013.20190521085533187\DICOMDIR'
    # show_3d_dicom(PathDicom)
    # show_dicom_dir(dicom_dir_path)

    # dicom_dir = r'.\hospital_data\3d\GENERAL\1.2.528.1.1001.200.2033.2157.206915881851013.20190521085533187\SDY00000\SRS00000'
    # png_dir = r'.\hospital_data\2d\dicom_to_png'
    # dicm_to_png(dicom_dir, png_dir)

    # path = r'.\data\Images\2-IM101.png' # (512, 512)
    # path = r'.\hospital_data\2d\dicom_to_png\0000_i.png' # (256, 256)
    # path = r'.\hospital_data\2d\dicom_to_png\19.png'
    # path = r'.\hospital_data\2d\Image(split_by_people&class)\1\clear_bladder\1 (3).png'
    path = r'.\hospital_data\dicom_to_png\52.png'
    show_png(path)

    # in_path = r'.\hospital_data\3d\GENERAL\1.2.528.1.1001.200.2033.2157.206915881851013.20190521085533187\SDY00000\SRS00000\IMG00000.dcm'
    # out_path = '.\output.png'
    # dcm_convert_to_png(in_path, out_path)

    # dicom_dir = r".\hospital_data\3d\GENERAL\1.2.528.1.1001.200.2033.2157.206915881851013.20190521085533187\SDY00000" 
    # dicom_to_png(dicom_dir)

    
