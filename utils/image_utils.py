import cv2
from PIL import Image
import numpy as np
import os
from config import IMG_EXTENSIONS

def pil_from_cv2(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def cv2_from_pil(image):
    image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_paths(data_dir, pattern='Albedo'):
    """
    get the paths all image files contains a certain pattern
    Args:
        data_dir:   data_dir contains images in it or in sub dirs of it
        pattern:    the pattern used to filter out images

    Returns:
        image_paths: a list contains paths of images

    """

    assert os.path.isdir(data_dir), '%s is not a valid derectory' % data_dir

    image_paths = []

    for root, dnames, fnames in sorted(os.walk(data_dir, followlinks=True)):
        for fname in fnames:
            if pattern in fname and is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)

    return image_paths


def get_train_val_paths(data_dir, split_ratio=0.2):
    all_paths = get_image_paths(data_dir, pattern='Albedo')
    data_size = len(all_paths)
    indices = np.random.permutation(data_size)
    val_split = int(data_size * split_ratio)

    val_paths = [all_paths[i] for i in indices[:val_split]]
    train_paths = [all_paths[i] for i in indices[val_split:]]

    return train_paths, val_paths


if __name__ == '__main__':
    data_dir = "dataset/biglook/"
    train_paths, test_paths = get_train_val_paths(data_dir, split_ratio=0.1)
    print(train_paths)
    print(len(train_paths))
    print(test_paths)
    print(len(test_paths))
