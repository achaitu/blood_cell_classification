# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 08:34:17 2018

@author: jeevan
"""

import numpy as np
import cv2
import os

BASE_DIR = 'H:/Projects/Blood_Classification/My_Work/this_sem_work/ALL_IDB_CLassification/'


def get_data(folder):
    X = []
    y = []

    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            for image_filename in os.listdir(folder + wbc_type):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_file = cv2.resize(img_file, (160,120))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(wbc_type)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

Image,Image_class = get_data(BASE_DIR +'/ALL_IDB/Test_Classes/')
for i in range(len(Image)):
    img = Image[i]
    ir = img[:,:,0]
    ib = img[:,:,2]
    im = ir-ib
    kernel = np.ones((2,2), np.uint8)
    img_erosion = cv2.dilate(im, kernel, iterations=2)
    kernel = np.ones((15,15), np.uint8)
    img_erosion = cv2.morphologyEx(img_erosion, cv2.MORPH_CLOSE, kernel)
    _,processed_image = cv2.threshold(img_erosion,140,255,cv2.THRESH_BINARY_INV)
    vals = ~processed_image.copy() 
    _,contours, hierarchy=cv2.findContours(vals, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    min_area, max_area = 5000, len(img)*len(img.T)/2
    #while i <= len(contours)-1
    #for i in range(len(contours)):
    #    area = cv2.contourArea(contours[i])
    #    if (area < min_area) or (area > max_area) :
    #        contours.remove(contours[i])
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]
    cv2.imwrite(Image_class[i]+str(i)+'.jpg',(img - out))
