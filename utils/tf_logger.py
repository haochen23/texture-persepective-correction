import numpy as np
from io import BytesIO
from datetime import datetime
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class TFLogger(object):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir + datetime.now().strftime("%Y-%m-%d %H-%M-%S"))

    def scalar_summary(self, name, value, step):
        """Log a scalar"""
        self.writer.add_scalar(name, value, step)

    def images_summary(self, name, images, step):
        """Log a list of images
        Inputs:
            images: torch tensor or numpy array
        """
        self.writer.add_images(name, images, step)
