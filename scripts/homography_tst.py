import cv2
import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt
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

tx, ty = 0,0
translation_matrix = np.array([[1, 0, tx],
                               [0, 1, ty],
                               [p1, p2, 1 ]])


H = np.dot(np.dot(rotation_matrix, shear_matrix), translation_matrix)

img = cv2.imread('images/1.png')

H_inv = inv(H)


img_distorted = cv2.warpPerspective(img, H, (img.shape[0], img.shape[1]))

# img = cv2.resize(img, (2048, 2048))

# img_distorted2 = cv2.warpPerspective(img, H, (img.shape[0], img.shape[1]))



img_distorted_large = cv2.resize(img_distorted, (2048, 2048))

img_restored = cv2.warpPerspective(img_distorted, H_inv, (img.shape[0], img.shape[1]))
scale = 2
H_inv[2, 0] /= scale
H_inv[2, 1] /= scale
H_inv[0, 2] *= scale
H_inv[1, 2] *= scale
img_restored_large = cv2.warpPerspective(img_distorted_large, H_inv, (img_distorted_large.shape[0], img_distorted_large.shape[1]))

plt.imshow(img_restored);plt.show()
plt.figure;plt.imshow(img_restored_large);plt.show()

final = cv2.resize(img_restored_large, (300, 300))

cv2.imshow('orig', img)
cv2.imshow('distorted', img_distorted)
cv2.imshow('restored', img_restored)
cv2.imshow('distorted_large', img_distorted_large)
cv2.imshow('restored_large', img_restored_large)
cv2.imshow("final", final)

cv2.waitKey(0)
cv2.destroyAllWindows()



image1 = image
image2= image.resize((800,800))

image1Orig = image1.copy()
image2Orig = image2.copy()

image1 = crop_least_square(image1)
image2 = crop_least_square(image2)

image1 = self.transform(image1)
image2 = self.transform(image2)

image1 = image1.view(1, 3, homography_config['image_height'], homography_config['image_width'])
image2 = image2.view(1, 3, homography_config['image_height'], homography_config['image_width'])

image1 = image1.cuda()
image2 = image2.cuda()

output1  = self.model(image1).squeeze()
output2  = self.model(image2).squeeze()

imageOrig = image2Orig.resize((512,512))


H_inv1 = decode_output(output1,
                              width=homography_config['load_width'],
                              height=homography_config['load_height'])

H_inv2 = decode_output(output2,
                              width=homography_config['load_width'],
                              height=homography_config['load_height'])
image1 = tensor2im(image1, imtype=np.uint8, normalize=True, tile=False)

image2 = tensor2im(image2, imtype=np.uint8, normalize=True, tile=False)


corrected_image1 = Image.fromarray(cv2.warpPerspective(np.array(image),
                                                              H_inv,
                                                              (image.width, image.height)))

image = image.resize((3000, 3000))
corrected_image2 = Image.fromarray(cv2.warpPerspective(np.array(image),
                                                              H_inv,
                                                              (image.width, image.height)))