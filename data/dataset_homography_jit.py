import os
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from utils.homography_utils import get_homography, decode_output, center_crop
from config import homography_config
from utils.image_utils import cv2_from_pil, pil_from_cv2


class HomographyDatasetJIT(data.Dataset):

    def __init__(self, image_paths, transform,
                 out_t_len=3,
                 load_width=homography_config['load_width'],
                 load_height=homography_config['load_height'],
                 out_width=homography_config['image_width'],
                 out_height=homography_config['image_height']
                 ):
        """
        Constructor of HomographyDatasetJIT
        Args:
            image_paths:    image paths, list
            transform:      transforms to apply
            out_t_len:      the tensor length of target
            load_width:     load image width
            load_height:    load image height
            out_width:      output image width
            out_height:     output image height
        """

        self.image_paths = image_paths
        self.load_width = load_width
        self.load_height = load_height
        self.out_width = out_width
        self.out_height = out_height
        self.out_t_len = out_t_len
        self.transform = transform

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        OriImg = Image.open(image_path).resize((self.load_width, self.load_height))
        if OriImg.mode == 'RGBA':
            OriImg = OriImg.convert('RGB')

        OriImg = cv2_from_pil(OriImg)
        target = torch.rand(self.out_t_len)
        H = get_homography(t=target, width=self.load_width, height=self.load_height)
        distortedImg = cv2.warpPerspective(OriImg, H, (self.load_width, self.load_height))

        croppedImg = center_crop(distortedImg,
                                 new_height=self.out_height,
                                 new_width=self.out_width)
        croppedImg = pil_from_cv2(croppedImg)

        if self.transform is not None:
            transformed_image = self.transform(croppedImg)
        else:
            transformed_image = croppedImg

        return transformed_image, target

    def __len__(self):
        """length of the dataset"""
        return len(self.image_paths)


if __name__ == '__main__':

    from data import get_homography_jit_loader
    from utils.image_utils import get_train_val_paths
    from config import homography_config
    data_dir = 'dataset/biglook/'

    train_paths, val_paths = get_train_val_paths(data_dir=data_dir,
                                                 split_ratio=homography_config['validation_split_ratio'])


    train_loader = get_homography_jit_loader(image_paths=train_paths,
                                             batch_size=5,
                                             out_t_len=3)

    val_loader = get_homography_jit_loader(image_paths=val_paths,
                                           batch_size=5,
                                           out_t_len=3)

    print(len(train_loader), len(val_loader))
    train_images, train_targets = next(iter(train_loader))
    val_images, val_targets = next(iter(val_loader))

    print(train_images.shape, train_targets.shape)
    print(val_images.shape, val_targets.shape)



















