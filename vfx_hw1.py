import numpy as np
import cv2
import glob
import os
import os.path as osp
import random

dirname = 'Memorial_SourceImages'
images, images_grayscale = [], []

for filename in np.sort(os.listdir(dirname)):
    if osp.splitext(filename)[1] in ['.png', '.jpg']:
        img = cv2.imread(osp.join(dirname, filename))
        images.append(img)
        img = cv2.imread(osp.join(dirname, filename), cv2.IMREAD_GRAYSCALE)
        images_grayscale.append(img)

#for i in range(num):
#    filename = "gray" + str(i) + ".png"
#    cv2.imwrite(filename, images_grayscale[i])

def translation(x, y):
    Matrix = np.float32([[1, 0, x],[0, 1, y]])
    return Matrix

def median_img(img):
    median = np.median(img)
    median_img = np.ones(img.shape)
    median_img[np.where(img < median)] = 0
    return median_img

def get_shift_vector(src, trg, x, y, threshold=4):
    h, w = trg.shape[:2]
    new_tx, new_ty = 0, 0
    min_distinct_pixel = np.inf

    median = np.median(src)
    median_src = median_img(src)
    median_trg = median_img(trg)

    ignore_pixels = np.ones(src.shape)
    ignore_pixels[np.where(np.abs(src - median) <= threshold)] = 0


    for fx in range(-1, 2):
        for fy in range(-1, 2):
            tmp_src = cv2.warpAffine(src, translation(x + fx, y + fy), (w, h))
            distinct_pixel = np.sum(np.logical_xor(tmp_src, trg) * ignore_pixels)
            if distinct_pixel < min_distinct_pixel:
                min_distinct_pixel = distinct_pixel
                new_fx, new_fy = fx, fy
		    
    return x + new_fx, y + new_fy

def get_image_alignment_vector(src, trg, depth=6):
    if depth == 0:
        fx, fy = get_shift_vector(src, trg, 0, 0)
        
    else:
        h, w = src.shape[:2]
        half_src = cv2.resize(src, (w//2, h//2))
        half_trg = cv2.resize(trg, (w//2, h//2))
        prev_fx, prev_fy = get_image_alignment_vector(half_src, half_trg, depth - 1)
        fx, fy = get_shift_vector(src, trg, prev_fx * 2, prev_fy * 2)

    return fx, fy
    
def image_align(src, trg, depth=6):
    h, w = trg.shape[:2]
    fx, fy = self.get_image_alignment_vector(src, trg, depth)
    return cv2.warpAffine(src, translation(fx, fy), (w, h))
    
