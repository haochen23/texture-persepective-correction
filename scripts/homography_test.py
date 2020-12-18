import cv2
import numpy as np
from numpy.linalg import inv
import math
theta = (np.random.rand() * 2 - 1) / 10
H = np.random.randn(3,3)
p1 = (np.random.rand() * 2 - 1) / 1000
p2 = (np.random.rand() * 2 - 1) / 1000
tx = (np.random.rand() * 2 - 1) * 50
ty = (np.random.rand() * 2 - 1) * 50

rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                     [math.sin(theta),  math.cos(theta), 0],
                     [0,            0,         1]])


shear_matrix = np.array([[1.0, 0, 0.0],
                            [0.0, 1, 0.0],
                            [0.0, 0.0, 1.0]])

tx, ty = 10,20
translation_matrix = np.array([[1, 0, tx],
                               [0, 1, ty],
                               [0, 0, 1 ]])

H = np.dot(np.dot(rotation_matrix, shear_matrix), translation_matrix)

img = cv2.imread('images/1.png')
img = cv2.resize(img, (300, 300))
H_1 = inv(H)
img_distorted = cv2.warpPerspective(img, H, (img.shape[0], img.shape[1]))

img_restored = cv2.warpPerspective(img_distorted, H_1, (img.shape[0], img.shape[1]))

cv2.imshow('orig', img)
cv2.imshow('distorted', img_distorted)
cv2.imshow('restored', img_restored)

cv2.waitKey(0)
cv2.destroyAllWindows()


