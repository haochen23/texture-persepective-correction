import math
import numpy as np


def distortionParameter(types):
    """
    distortionParameter() randomly generate distortion parameters which can
    be used to generate the corresponding flow map.
    We only care about rotation, perspective distortion.
    :param types: type of distortion 'rotation' or 'projective'
    :return:
        parameters: the corresponding distortion parameter
    """
    parameters = []

    if (types == 'rotation'):
        theta = np.random.random_sample() * 40 - 20
        radian = math.pi * theta / 180
        sina = math.sin(radian)
        cosa = math.cos(radian)
        parameters.append(sina)
        parameters.append(cosa)
        return parameters

    elif (types == 'projective'):

        x1 = 0
        x4 = np.random.random_sample() * 0.1 + 0.1

        x2 = 1 - x1
        x3 = 1 - x4

        y1 = 0.005
        y4 = 1 - y1
        y2 = y1
        y3 = y4

        a31 = ((x1 - x2 + x3 - x4) * (y4 - y3) - (y1 - y2 + y3 - y4) * (x4 - x3)) / ((x2 - x3) * (y4 - y3) - (x4 - x3) * (y2 - y3))
        a32 = ((y1 - y2 + y3 - y4) * (x2 - x3) - (x1 - x2 + x3 - x4) * (y2 - y3)) / ((x2 - x3) * (y4 - y3) - (x4 - x3) * (y2 - y3))

        a11 = x2 - x1 + a31 * x2
        a12 = x4 - x1 + a32 * x4
        a13 = x1

        a21 = y2 - y1 + a31 * y2
        a22 = y4 - y1 + a32 * y4
        a23 = y1

        parameters.append(a11)
        parameters.append(a12)
        parameters.append(a13)
        parameters.append(a21)
        parameters.append(a22)
        parameters.append(a23)
        parameters.append(a31)
        parameters.append(a32)
        return parameters


def distortionModel(types, xd, yd, width, height, parameter):
    if types == 'rotation':
        sina = parameter[0]
        cosa = parameter[1]
        xu = cosa * xd + sina * yd + (1 - sina - cosa) * width / 2
        yu = -sina * xd + cosa * yd + (1 + sina - cosa) * height / 2
        return xu, yu

    elif types == 'projective':
        a11 = parameter[0]
        a12 = parameter[1]
        a13 = parameter[2]
        a21 = parameter[3]
        a22 = parameter[4]
        a23 = parameter[5]
        a31 = parameter[6]
        a32 = parameter[7]
        im = xd/(width - 1.0)
        jm = yd/(height - 1.0)
        xu = (width - 1.0) *(a11*im + a12*jm +a13)/(a31*im + a32*jm + 1)
        yu = (height - 1.0)*(a21*im + a22*jm +a23)/(a31*im + a32*jm + 1)
        return xu, yu
