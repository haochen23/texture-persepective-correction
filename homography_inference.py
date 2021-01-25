from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import cv2
from utils.homography_utils import (crop_least_square,
                                    decode_output,
                                    pad_and_crop_to_size)
from config import homography_config


class HomographyInference:
    def __init__(self, model_path, transform):
        self.model = torch.load(model_path)
        self.transform = transform
        self.use_GPU = torch.cuda.is_available()
        if self.use_GPU:
            self.model.cuda()
        self.model.eval()

    def inference(self, image, out_dir='out/', out_file='result.png'):
        """
        inference method
        Args:
            image: PIL image object, original Image to be corrected
            out_dir: output dir
            out_file: output file name

        Returns:
            corrected image
        """

        imageOrig = image.copy()
        image = crop_least_square(image)
        # get the least square dimension
        ls_width, ls_height = image.size
        assert ls_width == ls_height
        image = self.transform(image)
        if self.use_GPU:
            image = image.cuda()
        image = image.view(1, 3, homography_config['image_height'], homography_config['image_width'])
        image = Variable(image)
        with torch.no_grad():
            output = self.model(image).squeeze()

        # obtain the inverse homography matrix from the output tensor
        H_inv = decode_output(output,
                              width=homography_config['image_width'],
                              height=homography_config['image_height'])

        padded_image = pad_and_crop_to_size(imageOrig, to_size=ls_height * 2)
        corrected_image = Image.fromarray(cv2.warpPerspective(padded_image,
                                                              H_inv,
                                                              (padded_image.shape[0], padded_image.shape[1])))
        return corrected_image
