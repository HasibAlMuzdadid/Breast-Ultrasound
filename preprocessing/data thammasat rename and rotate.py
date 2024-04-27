""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2024, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""



import os, glob
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
import cv2


# # Benign

# In[68]:


# Sort the list of filenames
# filenames = sorted(os.listdir(folder_path))
sorted_benign_images = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat 180/Image/Benign/"))
sorted_benign_masks = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat 180/Mask/Benign/"))


# In[69]:


# Load the images and masks from the folder
benign_images = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat 180/Image/Benign/", img)) for img in sorted_benign_images]
benign_masks = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat 180/Mask/Benign/", msk)) for msk in sorted_benign_masks]


# In[70]:


files = glob.glob("D:/Breast Cancer/Thammasat/Thammasat 180/Image/Benign/*") 
files_m = glob.glob("D:/Breast Cancer/Thammasat/Thammasat 180/Mask/Benign/*")
files = [file.split("Benign\\")[-1][:-4] for file in files]
files_m = [file.split("Benign\\")[-1][:-4] for file in files_m]


# In[71]:


print(f"{len(files)} , {len(files_m)}")
files[94], files_m[94]


# In[72]:


b0_images = []
b0_masks = []

b0_img_dir = "D:/Breast Cancer/Thammasat/Thammasat 270/Image/Benign"
b0_msk_dir = "D:/Breast Cancer/Thammasat/Thammasat 270/Mask/Benign"

for i in range(len(benign_images)):
    
    img = benign_images[i]
    #img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    b0_images.append(img)
    imsave(os.path.join(b0_img_dir, f"benign270 ({i+1}).png"), img)
    
    mask = benign_masks[i]
    #mask = cv2.resize(mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    b0_masks.append(mask)
    imsave(os.path.join(b0_msk_dir, f"benign270 ({i+1})_mask.png"), mask)


# In[73]:


for i in range(10):
    image = b0_images[i]
    mask = b0_masks[i]
    
    # Display the image and the true mask
    fig, (ax1,ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_axis_off()
    ax1.set_title("Image")
    ax2.imshow(mask)
    ax2.set_axis_off()
    ax2.set_title("Mask")

    plt.show()


# # Malignant

# In[40]:


sorted_malignant_images = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat 180/Image/Malignant/"))
sorted_malignant_masks = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat 180/Mask/Malignant/"))


# In[41]:


malignant_images = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat 180/Image/Malignant/", img)) for img in sorted_malignant_images]
malignant_masks = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat 180/Mask/Malignant/", msk)) for msk in sorted_malignant_masks]


# In[42]:


files = glob.glob("D:/Breast Cancer/Thammasat/Thammasat 180/Image/Malignant/*") 
files_m = glob.glob("D:/Breast Cancer/Thammasat/Thammasat 180/Mask/Malignant/*")
files = [file.split("Malignant\\")[-1][:-4] for file in files]
files_m = [file.split("Malignant\\")[-1][:-4] for file in files_m]


# In[43]:


print(f"{len(files)} , {len(files_m)}")
files[90], files_m[90]


# In[44]:


m0_images = []
m0_masks = []

m0_img_dir = "D:/Breast Cancer/Thammasat/Thammasat 270/Image/Malignant"
m0_msk_dir = "D:/Breast Cancer/Thammasat/Thammasat 270/Mask/Malignant"


for i in range(len(malignant_images)):
    
    img = malignant_images[i]
    #img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    m0_images.append(img)
    imsave(os.path.join(m0_img_dir, f"malignant270 ({i+1}).png"), img)
    
    mask = malignant_masks[i]
    #mask = cv2.resize(mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    m0_masks.append(mask)
    imsave(os.path.join(m0_msk_dir, f"malignant270 ({i+1})_mask.png"), mask)
    


# In[45]:


for i in range(10):
    image = m0_images[i]
    mask = m0_masks[i]
    
    # Display the image and the true mask
    fig, (ax1,ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_axis_off()
    ax1.set_title("Image")
    ax2.imshow(mask)
    ax2.set_axis_off()
    ax2.set_title("Mask")

    plt.show()


# # Generated

# In[5]:


sorted_malignantgan_images = sorted(os.listdir("C:/Users/THIS PC/Downloads/Thammasat Malignant GAN/"))


# In[6]:


malignantgan_images = [cv2.imread(os.path.join("C:/Users/THIS PC/Downloads/Thammasat Malignant GAN/", img)) for img in sorted_malignantgan_images]


# In[9]:


files = glob.glob("C:/Users/THIS PC/Downloads/Thammasat Malignant GAN/*") 
files = [file.split("GAN\\")[-1][:-4] for file in files]


# In[10]:


print(len(files))
files[90]


# In[11]:


mg_images = []
mg_img_dir = "D:/Breast Cancer/Thammasat/CycleGAN"


for i in range(len(malignantgan_images)):
    
    img = malignantgan_images[i]
    mg_images.append(img)
    imsave(os.path.join(mg_img_dir, f"malignant generated ({i+1}).png"), img)
    

