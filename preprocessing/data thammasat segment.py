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

# In[2]:


# Sort the list of filenames
# filenames = sorted(os.listdir(folder_path))
sorted_benign_images = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Image/Benign/"))
sorted_benign_masks = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Mask/Benign/"))


# In[3]:


# Load the images and masks from the folder
benign_images = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Image/Benign/", img)) for img in sorted_benign_images]
benign_masks = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Mask/Benign/", msk)) for msk in sorted_benign_masks]


# In[4]:


files = glob.glob("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Image/Benign/*") 
files_m = glob.glob("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Mask/Benign/*")
files = [file.split("Benign\\")[-1][:-4] for file in files]
files_m = [file.split("Benign\\")[-1][:-4] for file in files_m]


# In[6]:


print(f"{len(files)} , {len(files_m)}")
files[294], files_m[294]


# In[7]:


b_images = []
b_masks = []
b_segmented_images = []

b_seg_dir = "D:/Breast Cancer/Thammasat/Thammasat all + segmented/Segmented/Benign"

for i in range(len(benign_images)):
    
    img = benign_images[i]
    img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    b_images.append(img)
    
    mask = benign_masks[i]
    mask = cv2.resize(mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    b_masks.append(mask)
    
    result = img.copy()
    result[mask == 0] = 0
    result[mask != 0] = img[mask != 0]
    b_segmented_images.append(result)
    imsave(os.path.join(b_seg_dir, files[i] + "_segmented.png"), result)


# In[9]:


for i in range(10):
    image = b_images[i]
    mask = b_masks[i]
    segmented = b_segmented_images[i]
    
    # Display the image and the true mask
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
    ax1.imshow(image)
    ax1.set_axis_off()
    ax1.set_title("Image")
    ax2.imshow(mask)
    ax2.set_axis_off()
    ax2.set_title("Mask")
    ax3.imshow(segmented)
    ax3.set_axis_off()
    ax3.set_title("Segmented Mass")

    plt.show()
    fig.savefig(f"D:/Breast Cancer/Thammasat/Thammasat all + segmented/Example/Benign/{i+1}_segmented_result.png")


# # Malignant

# In[14]:


sorted_malignant_images = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Image/Malignant/"))
sorted_malignant_masks = sorted(os.listdir("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Mask/Malignant/"))


# In[15]:


malignant_images = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Image/Malignant/", img)) for img in sorted_malignant_images]
malignant_masks = [cv2.imread(os.path.join("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Mask/Malignant/", msk)) for msk in sorted_malignant_masks]


# In[16]:


files = glob.glob("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Image/Malignant/*") 
files_m = glob.glob("D:/Breast Cancer/Thammasat/Thammasat augmentation + synthetic/Mask/Malignant/*")
files = [file.split("Malignant\\")[-1][:-4] for file in files]
files_m = [file.split("Malignant\\")[-1][:-4] for file in files_m]


# In[17]:


print(f"{len(files)} , {len(files_m)}")
files[94], files_m[94]


# In[18]:


m_images = []
m_masks = []
m_segmented_images = []

m_seg_dir = "D:/Breast Cancer/Thammasat/Thammasat all + segmented/Segmented/Malignant"


for i in range(len(malignant_images)):
    
    img = malignant_images[i]
    img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    m_images.append(img)
    
    mask = malignant_masks[i]
    mask = cv2.resize(mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    m_masks.append(mask)
    
    result = img.copy()
    result[mask == 0] = 0
    result[mask != 0] = img[mask != 0]
    m_segmented_images.append(result)
    imsave(os.path.join(m_seg_dir, files[i] + "_segmented.png"), result)


# In[20]:


for i in range(10):
    image = m_images[i]
    mask = m_masks[i]
    segmented = m_segmented_images[i]
    
    # Display the image and the true mask
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
    ax1.imshow(image)
    ax1.set_axis_off()
    ax1.set_title("Image")
    ax2.imshow(mask)
    ax2.set_axis_off()
    ax2.set_title("Mask")
    ax3.imshow(segmented)
    ax3.set_axis_off()
    ax3.set_title("Segmented Mass")

    plt.show()
    fig.savefig(f"D:/Breast Cancer/Thammasat/Thammasat all + segmented/Example/Malignant/{i+1}_segmented_result.png")

