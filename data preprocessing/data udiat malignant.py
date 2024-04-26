""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2024, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""



# !pip install opencv-python


# In[1]:


import os
import numpy as np
import cv2

image_rows = 256
image_cols = 256

def create_train_data(data_path, name):
    train_data_path = os.path.join(data_path, "Image")
    train_mask_path = os.path.join(data_path, "Mask")    
    images = os.listdir(train_data_path)
    masks = os.listdir(train_mask_path)
    total = len(images)
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print(f"{'-'*30} creating training images {'-'*30}")
    for j in range(len(images)):

        img = cv2.imread(os.path.join(train_data_path, images[j]))
        img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
        #enhancement
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img_eqhist=cv2.equalizeHist(gray_img)
        img = cv2.cvtColor(gray_img_eqhist, cv2.COLOR_GRAY2BGR)
        #end enhancement
        img_mask = cv2.imread(os.path.join(train_mask_path, masks[j]), 0)
        img_mask = cv2.resize(img_mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        if i % 100 == 0:
            print(f"Done: {i}/{total} images")
        i += 1
    print("Loading done")


    np.save(os.path.join("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant", "imgs_train_"+name+".npy"), imgs)
    np.save(os.path.join("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant", "imgs_mask_train_"+name+".npy"), imgs_mask)
    print("Saving to .npy files done")


# In[2]:


create_train_data("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant/train", "malignant")


# In[3]:


def load_train_data(name):
    imgs_train = np.load("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant/imgs_train_"+name+".npy")
    imgs_mask_train = np.load("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant/imgs_mask_train_"+name+".npy")
    return imgs_train, imgs_mask_train


# In[4]:


img,mask=load_train_data("malignant")
img.shape,mask.shape


# In[5]:


def create_test_data(data_path2, name):
    test_data_path = os.path.join(data_path2, "Image")
    test_mask_path = os.path.join(data_path2, "Mask")    
    images = os.listdir(test_data_path)
    masks = os.listdir(test_mask_path)
    total = len(images)
    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print(f"{'-'*20} creating testing images {'-'*20}")
    for j in range(len(images)):

        img = cv2.imread(os.path.join(test_data_path, images[j]))
        img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
        #enhancement
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_img_eqhist=cv2.equalizeHist(gray_img)
        img = cv2.cvtColor(gray_img_eqhist, cv2.COLOR_GRAY2BGR)
        #end enhancement
        img_mask = cv2.imread(os.path.join(test_mask_path, masks[j]), 0)
        img_mask = cv2.resize(img_mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        if i % 100 == 0:
            print(f"Done: {i}/{total} images")
        i += 1
    print("Loading done")


    np.save(os.path.join("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant", "imgs_test_"+name+".npy"), imgs)
    np.save(os.path.join("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant", "imgs_id_test_"+name+".npy"), imgs_mask)
    print("Saving to .npy files done")


# In[6]:


create_test_data("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant/test", "malignant")


# In[7]:


def load_test_data(name):
    imgs_test = np.load("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant/imgs_test_"+name+".npy")
    imgs_id = np.load("D:/Breast Cancer/UDIAT Dataset/UDIAT traditional augmentation Malignant/imgs_id_test_"+name+".npy")
    return imgs_test, imgs_id


# In[8]:


img,mask=load_test_data("malignant")
img.shape,mask.shape

