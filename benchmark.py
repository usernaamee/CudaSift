import cv2
import pycusift
import numpy as np
from time import time
import matplotlib.pyplot as plt

# uncomment both @profile when running with python3 -m memory_profiler benchmark.py
# @profile
def cuda_func(orig_img):
    img = orig_img.astype('float32'); 
    start = time()
    tmp = pycusift.sift_feature_extractor(img)
    end = time()
    print('pycusift time: {:.3f}'.format(end - start))
    tmp = np.array(tmp).reshape((-1, 130))
    print('Total features detected:', len(tmp))
    kp = []
    des = []
    for entry in tmp:
        kp_obj = cv2.KeyPoint(entry[0], entry[1], 0)
        des_obj = entry[2:]
        kp.append(kp_obj)
        des.append(des_obj)
    outimg = np.zeros_like(img)
    outimg1 = cv2.drawKeypoints(orig_img, kp, outimg)
    return outimg1


# @profile
def opencv_func(orig_img):
    img = orig_img.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    start = time()
    kp, des = sift.detectAndCompute(img, None)
    end = time()
    print('opencv time: {:.3f}'.format(end - start))
    print('Total features detected:', len(kp))
    outimg = np.zeros_like(img)
    outimg2 = cv2.drawKeypoints(orig_img, kp, outimg)
    return outimg2


orig_img = cv2.imread('data/img1.png', 0)
outimg1 = cuda_func(orig_img)
outimg2 = opencv_func(orig_img)


plt.subplot(121)
plt.imshow(outimg1)
plt.subplot(122)
plt.imshow(outimg2)
plt.show()
