import cv2
from PIL import Image
import numpy as np
import os
from utils.image_utils import cv2_from_pil, pil_from_cv2, get_image_paths, get_train_val_paths, is_image_file
from config import IMG_EXTENSIONS

def test_cv2_from_pil():
    image_rgb = Image.open("images/1.png").convert('RGB')
    image_gray = image_rgb.convert('L')

    cv2_image_bgr = cv2_from_pil(image_rgb)
    cv2_image_gray = cv2_from_pil(image_gray)

    assert len(cv2_image_bgr.shape) == 3
    assert len(cv2_image_gray.shape) == 2
    assert (cv2_image_bgr[:,:,::-1] == np.array(image_rgb)).all()
    assert (cv2_image_gray == np.array(image_gray)).all()


def test_pil_from_cv2():

    cv2_image_bgr = cv2.imread("images/1.png")
    cv2_image_gray = cv2.cvtColor(cv2_image_bgr, cv2.COLOR_BGR2GRAY)

    image_rgb = pil_from_cv2(cv2_image_bgr)
    image_gray = pil_from_cv2(cv2_image_gray)

    assert image_rgb.mode == 'RGB'
    assert image_gray.mode == 'L'
    assert all(cv2_image_bgr[:, :, ::-1] == np.array(image_rgb))
    assert all(cv2_image_gray == np.array(image_gray))


def test_is_image_file():
    fake_image_files = ['fake_image' + ext for ext in IMG_EXTENSIONS]

    for image_file in fake_image_files:
        assert is_image_file(image_file)

    not_image_files = ['not_image' + ext for ext in ['.doc', '.docx', '.pdf', '.xls', '.py', '.mat']]

    for other_file in not_image_files:
        assert is_image_file(other_file) == False


def test_get_image_paths():
    data_dir = 'dataset/biglook/'
    image_paths = get_image_paths(data_dir=data_dir, pattern='Albedo')

    assert type(image_paths) == list
    assert len(image_paths) == 10
    for item in image_paths:
        assert os.path.exists(item)
        assert is_image_file(item)


def test_get_train_val_paths():
    data_dir = 'dataset/biglook/'
    split_ratio = 0.1

    train_paths, test_paths = get_train_val_paths(data_dir, split_ratio=split_ratio)

    assert type(train_paths) == list
    assert type(test_paths) == list
    # since there are 10 images in dataset/biglook/
    assert len(train_paths) == 9
    assert len(test_paths) == 1

    for item in train_paths:
        assert os.path.exists(item)
        assert is_image_file(item)

    for item in test_paths:
        assert os.path.exists(item)
        assert is_image_file(item)











