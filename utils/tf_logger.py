import tensorflow as tf
import numpy as np
from io import BytesIO
from datetime import datetime
from PIL import Image

class TFLogger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def scalar_summary(self, name, value, step):
        """log a scalar variable"""
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def image_summary(self, name, images, step):
        """Log a list of images"""

        for i, image in enumerate(images):
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            if image.mode == 'F':
                image = image.convert('L')
            image_bytes = BytesIO()
            image.save(image_bytes, format='PNG')
            with self.writer.as_default():
                tf.summary.image(name, image_bytes.getvalue(),step=step)





