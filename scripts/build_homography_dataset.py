import scipy.io as scio
import os
# os.chdir('C:/python_code/texture-persepective-correction')
print(os.getcwd())
import argparse
import cv2
import torch
import glob
from utils.homography_utils import get_homography, decode_output, center_crop
from config import homography_config
from numpy.linalg import inv



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, type=str, help='Path to the input directory containing subdirectories of source files')
parser.add_argument('--output_dir', required=True, type=str, help='Path to the output directory containing train and val subdirectories')
parser.add_argument('--split_ratio', type=float, default=0.1, help='how much data to put in val, default=0.1')
args = vars(parser.parse_args())


def generate_data(f_path, k, save_dir, width=512, height=512, out_len=5):
    """
    Generate data from source images
    Args:
        f_path:    file path to the image
        k:         count number for this image
        save_dir:  save directory
        width:
        height:
        out_len:

    Returns:

    """

    OriImg = cv2.imread(f_path)
    OriImg = cv2.resize(OriImg, (width, height))

    # obtaining training target
    target = torch.rand(out_len)
    H = get_homography(t=target, width=width, height=height)
    distortedImg = cv2.warpPerspective(OriImg, H, (height, width))
    croppedImg = center_crop(distortedImg, new_height=homography_config["image_height"], new_width=homography_config["image_width"])

    trainDisPath = save_dir + 'distorted/'
    trainTargetPath = save_dir + 'target/'
    if not os.path.exists(trainDisPath):
        os.makedirs(trainDisPath, exist_ok=True)
    if not os.path.exists(trainTargetPath):
        os.makedirs(trainTargetPath, exist_ok=True)

    saveImgPath = '%s%s%s%s%s%s' % (trainDisPath, '/', 'distorted', '_', str(k).zfill(6), '.jpg')
    saveMatPath = '%s%s%s%s%s%s' % (trainTargetPath, '/', 'distorted', '_', str(k).zfill(6), '.mat')
    target = target.detach().cpu().numpy()

    cv2.imwrite(saveImgPath, croppedImg)
    scio.savemat(saveMatPath, {'target': target})

    # showing the restored image just for validation purpose
    border_size = (homography_config['load_width'] - homography_config['image_width']) // 2
    bordered_cropped = cv2.copyMakeBorder(croppedImg,
                                          top=border_size,
                                          bottom=border_size,
                                          left=border_size,
                                          right=border_size,
                                          borderType=cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])
    H_inv = inv(H)
    restoredImg = cv2.warpPerspective(bordered_cropped,
                                      H_inv,
                                      (homography_config["load_height"],
                                       homography_config["load_width"]))

    # cv2.imshow('Distorted', croppedImg)
    # cv2.imshow("Restored", restoredImg)
    overlay = cv2.addWeighted(OriImg, 0.5, restoredImg, 0.5, 1)
    cv2.imshow('Overlay restored', overlay)
    cv2.waitKey(1)



def build_homography(input_dir, output_dir, split_ratio = 0.2):
    """build homography dataset from input_dir to output_dir
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
                generate_data(f_path=f_path, k=count,
                              save_dir=output_val,
                              width=homography_config['load_width'],
                              height=homography_config['load_height'])
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
                generate_data(f_path=f_path, k=count,
                              save_dir=output_train,
                              width=homography_config['load_width'],
                              height=homography_config['load_height'])
                count += 1

        except Exception as e:
            print(e)


if __name__ == '__main__':
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    split_ratio = args['split_ratio']
    build_homography(input_dir, output_dir, split_ratio=split_ratio)
    cv2.destroyAllWindows()



