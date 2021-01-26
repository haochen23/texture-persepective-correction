import numpy as np
import torch
import math
from numpy.linalg import inv
import cv2
from config import homography_config
from PIL import Image


def get_homography(t, width=512, height=512):
    """
    This function transforms a size-5 torch tensor to a homography matrix
    :param t: a size 5 torch tensor float32
           width:  the width of the image that the H should be applied to
           height: the height of the image that the H should be applied to

    :return: H: the resultant homography matrix, numpy array
    """
    array = t.detach().cpu().numpy()
    # rotation angle, rotate from center theta in range [-20, 20]
    theta = (array[0] * 2 - 1) / 40

    # perspective parameters
    p1 = (array[1] * 2 - 1) / 6000
    p2 = (array[2] * 2 - 1) / 6000

    if array.size == 5:
        # add x, y translation parameters, within 5% range
        tx = int((array[3] * 2 - 1) * width * homography_config['translation_range'])
        ty = int((array[4] * 2 - 1) * height * homography_config['translation_range'])
    elif array.size == 3:
        tx = 0
        ty = 0
    else:
        raise NotImplementedError(f"The size of tensor {array.size} is not implemented in get_homography function.")

    alpha = math.cos(theta * np.pi)
    beta = -math.sin(theta * np.pi)

    # rotation from centre, and x, y translation
    rotation_matrix = np.array([[alpha, beta, (1 - alpha) * width / 2 - beta * height / 2 + tx],
                                [-beta, alpha, beta * width / 2 + (1 - alpha) * height / 2 + ty],
                                [0, 0, 1]])

    # rotation from top left corner
    # rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
    #                             [math.sin(theta), math.cos(theta), 0],
    #                             [0, 0, 1]])

    shear_matrix = np.array([[1.0, 0, 0.0],
                             [0.0, 1, 0.0],
                             [0.0, 0.0, 1.0]])

    perspective_matrix = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [p1, p2, 1]])
    H = np.dot(np.dot(rotation_matrix, shear_matrix), perspective_matrix)

    return H


def decode_output(t, width=512, height=512):
    """
    This function decodes the output from the homography model
    :param t: a size 5 torch tensor float32
    :param width: should be same as in get_homography function
    :param height: should be same as in get_homography function
    :return:
        H_inv: decoded homography matrix that can be used for inference
    """

    H = get_homography(t=t, width=width, height=height)
    H_inv = inv(H)

    return H_inv


def center_crop(img, new_height=256, new_width=256):
    """
    Function to centre crop an image using opencv

    :param img: cv2 image, the shape should be larger than new_height and new_width
    :param new_height: resultant image height
    :param new_width: resultant image width
    :return:
        cropped_img: center cropped cv2 image with new_height and new_width
    """

    width, height = img.shape[:2]
    if width < new_width or height < new_height:
        print("Center Crop Warning: Input image shape smaller than Cropped image shape. Resizing it.")
        cropped_img = cv2.resize(img, (new_width, new_height))
    else:
        center = np.array(img.shape[:2]) / 2
        x = center[1] - new_width / 2
        y = center[0] - new_height / 2
        cropped_img = img[int(y):int(y + new_height), int(x):int(x + new_width)]

    return cropped_img


def crop_least_square(image):
    """
    Creates the least square centre-cropped image
    Args:
        image: PIL Image
    Returns:
        cropped least square image
    """
    if image.width > image.height:
        delta = int((image.width - image.height) / 2)
        return image.crop((delta, 0, image.height + delta, image.height))
    else:
        delta = int((image.height - image.width) / 2)
        return image.crop((0, delta, image.width, image.width + delta))


def pad_and_crop_to_size(image, to_size=1024):
    """
    pad image (or crop) to to_size
    Args:
        image: PIL image object, RGB
        to_size: target square image size

    Returns:

    """
    width, height = image.size
    if width >= to_size and height < to_size:
        # only need to add zeros at top and bottom
        margin = image.width - image.height
        top = int(margin / 2)
        new_image = Image.new(image.mode, (image.width, image.height + margin), (0, 0, 0))
        new_image.paste(image, (0, top))
        new_image = Image.fromarray(center_crop(np.array(new_image),
                                                new_height=to_size,
                                                new_width=to_size))
    elif width < to_size and height >= to_size:
        # only need to add zeros at left and right
        margin = image.height - image.width
        left = int(margin / 2)
        new_image = Image.new(image.mode, (image.width + margin, image.height), (0, 0, 0))
        new_image.paste(image, (left, 0))
        new_image = Image.fromarray(center_crop(np.array(new_image),
                                                new_height=to_size,
                                                new_width=to_size))
    elif width >= to_size and height >= to_size:
        # do no need to add zero paddings
        new_image = Image.fromarray(center_crop(np.array(image),
                                                new_height=to_size,
                                                new_width=to_size))
    else:
        # need to add zeros to left right top bottom
        margin_h = to_size - width
        margin_v = to_size - height
        top = int(margin_v / 2)
        left = int(margin_h /2)

        new_image = cv2.copyMakeBorder(np.array(image),
                                       borderType=cv2.BORDER_CONSTANT,
                                       top=top,
                                       bottom=top,
                                       left=left,
                                       right=left,
                                       value=[0, 0, 0])
        new_image = Image.fromarray(new_image)

    return new_image


if __name__ == '__main__':
    image = Image.open('images/1.png').convert('RGB')
    image = image.resize((1024, 1024))
    new_image = pad_and_crop_to_size(image)
    new_image.show()




