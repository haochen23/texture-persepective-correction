import os
import torch
from torch.utils import data
from torchvision import transforms
import scipy.io as spio
import numpy as np
from PIL import Image


class FlowNetDataset(data.Dataset):
    def __init__(self, imageDir, flowDir, transform):

        self.image_paths = []
        self.flow_paths = []

        for img in os.listdir(imageDir):
            self.image_paths.append(os.path.join(imageDir, img))

        for flow in os.listdir(flowDir):
            self.flow_paths.append(os.path.join(flowDir, flow))

        self.image_paths.sort()
        self.flow_paths.sort()

        self.transform = transform

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        flow_path = self.flow_paths[index]

        image = Image.open(image_path)
        flow = spio.loadmat(flow_path)

        flow_x = flow['u'].astype(np.float32)
        flow_y = flow['v'].astype(np.float32)

        flow_x = flow_x[np.newaxis, :]
        flow_y = flow_y[np.newaxis, :]

        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        return transformed_image, flow_x, flow_y

    def __len__(self):
        return len(self.flow_paths)


if __name__ == '__main__':
    from data import get_loader
    data_loader = get_loader('../dataset/processed/train/distorted/',
                             '../dataset/processed/train/flow/',
                             batch_size=5,
                             data_type="flow"
                             )
    print(len(data_loader))
    images, flow_xs, flow_ys = next(iter(data_loader))
    print(images.shape)
    print(flow_xs.shape)
    print(flow_ys.shape)
