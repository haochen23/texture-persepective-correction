from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os

from models.model_flownet import FlowNet
from utils.resampling import rectification


class FlowNetInference:
    def __init__(self, model_path, transform):
        self.model = torch.load(model_path)
        self.transform = transform
        self.use_GPU = torch.cuda.is_available()
        if self.use_GPU:
            self.model.cuda()
        self.model.eval()

    def inference(self, image_path, out_dir= 'out/', out_file='result.png'):
        image = Image.open(image_path).convert('RGB').resize((256, 256))
        image_copy = np.array(image)
        image = self.transform(image)
        if self.use_GPU:
            image = image.cuda()
        image = image.view(1, 3, 256, 256)
        image = Variable(image)
        with torch.no_grad():
            flow_output = self.model(image)
        flow = flow_output.data.cpu().numpy().squeeze()

        # perspective corrected image
        corrected_image, resMask = rectification(image_copy, flow)
        corrected_image = Image.fromarray(corrected_image)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        corrected_image.save(out_dir + out_file)
        return corrected_image


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    flownet_infer = FlowNetInference('best_model.pt', transform=transform)
    corrected_image = flownet_infer.inference('dataset/processed/val/distorted/projective_000009.jpg', out_dir='out/',
                                              out_file='result.png')


