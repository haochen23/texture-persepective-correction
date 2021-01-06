import numpy as np
import torch
import math
from numpy.linalg import inv


def get_homography(t, width=512, height=512):
    """
    This function transforms a size-3 torch tensor to a homography matrix
    :param t: a size 3 torch tensor float32
           width:  the width of the image that the H should be applied to
           height: the height of the image that the H should be applied to

    :return: H: the resultant homography matrix, numpy array
    """
    array = t.detach().cpu().numpy()
    # rotation angle, theta in range [-30, 30]
    theta = (array[0] * 2 - 1) / 60

    # perspective parameters
    p1 = (array[1] * 2 - 1) / 2000
    p2 = (array[2] * 2 - 1) / 2000

    alpha = math.cos(theta * np.pi)
    beta = -math.sin(theta * np.pi)
    rotation_matrix = np.array([[alpha, beta, (1 - alpha) * width/2 - beta * height/2],
                                [-beta, alpha, beta * width/2 + (1 - alpha) * height/2],
                                [0, 0, 1]])
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
    :param t: a size 3 torch tensor float32
    :param width: should be same as in get_homography function
    :param height: should be same as in get_homography function
    :return:
        H_inv: decoded homography matrix that can be used for inference
    """

    H = get_homography(t=t, width=width, height=height)
    H_inv = inv(H)

    return H_inv







