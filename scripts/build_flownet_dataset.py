import numpy as np
import skimage
import scipy.io as scio
import data.distortion_model as distortion_model
import os
import argparse
import cv2
import torch
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, type=str, help='Path to the input directory containing subdirectories of source files')
parser.add_argument('--output_dir', required=True, type=str, help='Path to the output directory containing train and val subdirectories')
parser.add_argument('--split_ratio', type=float, default=0.1, help='how much data to put in val, default=0.1')
args = vars(parser.parse_args())

def generate_rotation(f_path, k, save_dir):
    types = "rotation"
    width = 512
    height = 512

    parameters = distortion_model.distortionParameter(types)
    OriImg = cv2.imread(f_path)
    OriImg = cv2.resize(OriImg, (width, height))

    disImg = np.array(np.zeros(OriImg.shape), dtype=np.uint8)
    u = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)
    v = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)

    cropImg = np.array(np.zeros((int(height / 2), int(width / 2), 3)), dtype=np.uint8)
    crop_u = np.array(np.zeros((int(height / 2), int(width / 2))), dtype=np.float32)
    crop_v = np.array(np.zeros((int(height / 2), int(width / 2))), dtype=np.float32)

    # crop range
    xmin = int(width * 1 / 4)
    xmax = int(width * 3 / 4 - 1)
    ymin = int(height * 1 / 4)
    ymax = int(height * 3 / 4 - 1)

    for i in range(width):
        for j in range(height):

            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)

            if (0 <= xu < width - 1) and (0 <= yu < height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j

                # Bilinear interpolation
                Q11 = OriImg[int(yu), int(xu), :]
                Q12 = OriImg[int(yu), int(xu) + 1, :]
                Q21 = OriImg[int(yu) + 1, int(xu), :]
                Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]

                disImg[j, i, :] = Q11 * (int(xu) + 1 - xu) * (int(yu) + 1 - yu) + \
                                  Q12 * (xu - int(xu)) * (int(yu) + 1 - yu) + \
                                  Q21 * (int(xu) + 1 - xu) * (yu - int(yu)) + \
                                  Q22 * (xu - int(xu)) * (yu - int(yu))

                if (xmin <= i <= xmax) and (ymin <= j <= ymax):
                    cropImg[j - ymin, i - xmin, :] = disImg[j, i, :]
                    crop_u[j - ymin, i - xmin] = u[j, i]
                    crop_v[j - ymin, i - xmin] = v[j, i]

    trainDisPath = save_dir + 'distorted/'
    trainUvPath = save_dir + 'flow/'
    if not os.path.exists(trainDisPath):
        os.makedirs(trainDisPath, exist_ok=True)
    if not os.path.exists(trainUvPath):
        os.makedirs(trainUvPath, exist_ok=True)
    saveImgPath = '%s%s%s%s%s%s' % (trainDisPath, '/',types,'_', str(k).zfill(6), '.jpg')
    saveMatPath = '%s%s%s%s%s%s' % (trainUvPath, '/', types, '_', str(k).zfill(6), '.mat')
    cv2.imwrite(saveImgPath, cropImg)
    scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})
    cv2.imshow('rotation', cropImg)
    cv2.waitKey(1)


def generate_projection(f_path, k, save_dir):
    types = 'projective'
    width = 256
    height = 256

    parameters = distortion_model.distortionParameter(types)
    OriImg = cv2.imread(f_path)
    ScaImg = cv2.resize(OriImg, (width, height))
    ScaImg = skimage.img_as_ubyte(ScaImg)

    padImg = np.array(np.zeros((ScaImg.shape[0] + 1, ScaImg.shape[1] + 1, 3)), dtype=np.uint8)
    padImg[0:height, 0:width, :] = ScaImg[0:height, 0:width, :]
    padImg[height, 0:width, :] = ScaImg[height - 1, 0:width, :]
    padImg[0:height, width, :] = ScaImg[0:height, width - 1, :]
    padImg[height, width, :] = ScaImg[height - 1, width - 1, :]

    disImg = np.array(np.zeros(ScaImg.shape), dtype=np.uint8)
    u = np.array(np.zeros((ScaImg.shape[0], ScaImg.shape[1])), dtype=np.float32)
    v = np.array(np.zeros((ScaImg.shape[0], ScaImg.shape[1])), dtype=np.float32)

    for i in range(width):
        for j in range(height):

            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)

            if (0 <= xu <= width - 1) and (0 <= yu <= height - 1):
                u[j][i] = xu - i
                v[j][i] = yu - j

                # Bilinear interpolation
                Q11 = padImg[int(yu), int(xu), :]
                Q12 = padImg[int(yu), int(xu) + 1, :]
                Q21 = padImg[int(yu) + 1, int(xu), :]
                Q22 = padImg[int(yu) + 1, int(xu) + 1, :]

                disImg[j, i, :] = Q11 * (int(xu) + 1 - xu) * (int(yu) + 1 - yu) + \
                                  Q12 * (xu - int(xu)) * (int(yu) + 1 - yu) + \
                                  Q21 * (int(xu) + 1 - xu) * (yu - int(yu)) + \
                                  Q22 * (xu - int(xu)) * (yu - int(yu))

    trainDisPath = save_dir + 'distorted/'
    trainUvPath = save_dir + 'flow/'
    if not os.path.exists(trainDisPath):
        os.makedirs(trainDisPath, exist_ok=True)
    if not os.path.exists(trainUvPath):
        os.makedirs(trainUvPath, exist_ok=True)
    saveImgPath = '%s%s%s%s%s%s' % (trainDisPath, '/', types, '_', str(k).zfill(6), '.jpg')
    saveMatPath = '%s%s%s%s%s%s' % (trainUvPath, '/', types, '_', str(k).zfill(6), '.mat')
    cv2.imshow('projective', disImg)
    cv2.waitKey(1)
    cv2.imwrite(saveImgPath, disImg)
    scio.savemat(saveMatPath, {'u': u, 'v': v})



def build_dataset(input_dir, output_dir, split_ratio = 0.2):
    """build dataset from input_dir to output_dir
    Args:
        input_dir: containing subdirectories of images (downloaded from s3)
        output_dir: prepared dataset and split into train and val
    """
    output_train = os.path.join(output_dir, 'train/')
    output_val = os.path.join(output_dir, 'val/')

    if not os.path.exists(output_train):
        os.makedirs(output_train, exist_ok=True)
    if not os.path.exists(output_val):
        os.makedirs(output_val, exist_ok=True)

    subdirs = [os.path.join(input_dir, o) for o in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, o))]

    dataset_size = len(subdirs)
    indices = torch.randperm(dataset_size)
    val_indices = int(dataset_size * split_ratio)



    # build val
    if os.path.exists(output_val+'distorted/'):
        count = int(sorted(os.listdir(output_val+'distorted/'))[-1].split('.')[0].split('_')[-1]) + 1
    else:
        count = 0

    for subdir in [subdirs[i] for i in indices[:val_indices]]:
        try:
            files = glob.glob(subdir + '/*Albedo*')
            if len(files) > 0:
                f_path = files[0]
                generate_projection(f_path=f_path, k=count, save_dir=output_val)
                generate_rotation(f_path=f_path, k=count, save_dir=output_val)

                count += 1
        except Exception as e:
            print(e)


    # build train
    if os.path.exists(output_train + 'distorted/'):
        count = int(sorted(os.listdir(output_train + 'distorted/'))[-1].split('.')[0].split('_')[-1]) + 1
    else:
        count = 0
    for subdir in [subdirs[i] for i in indices[val_indices:]]:
        try:
            files = glob.glob(subdir + '/*Albedo*')
            if files is not None:
                f_path = files[0]
                generate_projection(f_path=f_path, k=count, save_dir=output_train)
                generate_rotation(f_path=f_path, k=count, save_dir=output_train)

                count += 1
        except Exception as e:
            print(e)


if __name__ == '__main__':
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    split_ratio = args['split_ratio']
    # input_dir = 'dataset/biglook/'
    # output_dir = 'dataset/processed2/'
    split_ratio = 0.2
    build_dataset(input_dir, output_dir, split_ratio=split_ratio)
    cv2.destroyAllWindows()