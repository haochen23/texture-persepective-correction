import os
import torch
from torch.utils import data
from torchvision import transforms
import scipy.io as spio
import numpy as np
from PIL import Image


class HomographyDataset(data.Dataset):
    def __init__(self, imageDir, targetDir, transform):
        self.image_paths = []
        self.target_paths = []

        for img in os.listdir(imageDir):
            self.image_paths.append(os.path.join(imageDir, img))

        for target in os.listdir(targetDir):
            self.target_paths.append(os.path.join(targetDir, target))

        self.image_paths.sort()
        self.target_paths.sort()

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target_path = self.target_paths[index]

        image = Image.open(image_path)
        target_mat= spio.loadmat(target_path)
        target = target_mat['target'].astype(np.float32)
        print(target)

        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        return transformed_image, target

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    from data import get_loader

    data_loader = get_loader('../dataset/homography/train/distorted/',
                             '../dataset/homography/train/target/',
                             batch_size=5,
                             data_type="homo"
                             )

    print(len(data_loader))
    images, targets = next(iter(data_loader))
    print(images.shape)
    print(targets.shape)