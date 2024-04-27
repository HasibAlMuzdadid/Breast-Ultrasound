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
sorted_benign_images = sorted(os.listdir("D:/Dataset/BUSI with Ground Truth Augmented/Image/benign/"))
sorted_benign_masks = sorted(os.listdir("D:/Dataset/BUSI with Ground Truth Augmented/Mask/benign/"))


# In[4]:


# Load the images and masks from the folder
benign_images = [cv2.imread(os.path.join("D:/Dataset/BUSI with Ground Truth Augmented/Image/benign/", img)) for img in sorted_benign_images]
benign_masks = [cv2.imread(os.path.join("D:/Dataset/BUSI with Ground Truth Augmented/Mask/benign/", msk)) for msk in sorted_benign_masks]


# In[ ]:


files = glob.glob("D:/Dataset/BUSI with Ground Truth Augmented/Image/benign/*.png") 
files = [file.split("benign\\")[-1][:-4] for file in files]


# In[42]:


b_images = []
b_masks = []
b_segmented_images = []

b_img_dir = "D:/Dataset/BUSI Segmented/Image/Benign"
b_msk_dir = "D:/Dataset/BUSI Segmented/Mask/Benign"
b_seg_dir = "D:/Dataset/BUSI Segmented/Segmented/Benign"

for i in range(len(benign_images)):
    
    img = benign_images[i]
    img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    b_images.append(img)
    imsave(os.path.join(b_img_dir, files[i] + ".png"), img)
    
    mask = benign_masks[i]
    mask = cv2.resize(mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    b_masks.append(mask)
    imsave(os.path.join(b_msk_dir, files[i] + "_mask.png"), mask)
    
    result = img.copy()
    result[mask == 0] = 0
    result[mask != 0] = img[mask != 0]
    b_segmented_images.append(result)
    imsave(os.path.join(b_seg_dir, files[i] + "_segmented.png"), result)


# In[47]:


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
    fig.savefig(f"D:/Dataset/BUSI Segmented/Example/Benign/{i+1}_segmented_result.png")


# # Malignant

# In[48]:


sorted_malignant_images = sorted(os.listdir("D:/Dataset/Malignant All/malignant all images/"))
sorted_malignant_masks = sorted(os.listdir("D:/Dataset/Malignant All/malignant all masks/"))


# In[49]:


malignant_images = [cv2.imread(os.path.join("D:/Dataset/Malignant All/malignant all images/", img)) for img in sorted_malignant_images]
malignant_masks = [cv2.imread(os.path.join("D:/Dataset/Malignant All/malignant all masks/", msk)) for msk in sorted_malignant_masks]


# In[54]:


files = glob.glob("D:/Dataset/Malignant All/malignant all images/*.png") 
files = [file.split("images\\")[-1][:-4] for file in files]


# In[56]:


m_images = []
m_masks = []
m_segmented_images = []

m_img_dir = "D:/Dataset/BUSI Segmented/Image/Malignant"
m_msk_dir = "D:/Dataset/BUSI Segmented/Mask/Malignant"
m_seg_dir = "D:/Dataset/BUSI Segmented/Segmented/Malignant"

for i in range(len(malignant_images)):
    
    img = malignant_images[i]
    img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    m_images.append(img)
    imsave(os.path.join(m_img_dir, files[i] + ".png"), img)
    
    mask = malignant_masks[i]
    mask = cv2.resize(mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    m_masks.append(mask)
    imsave(os.path.join(m_msk_dir, files[i] + "_mask.png"), mask)
    
    result = img.copy()
    result[mask == 0] = 0
    result[mask != 0] = img[mask != 0]
    m_segmented_images.append(result)
    imsave(os.path.join(m_seg_dir, files[i] + "_segmented.png"), result)


# In[57]:


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
    fig.savefig(f"D:/Dataset/BUSI Segmented/Example/Malignant/{i+1}_segmented_result.png")


# # Normal

# In[58]:


sorted_normal_images = sorted(os.listdir("D:/Dataset/Normal All/normal all images/"))
sorted_normal_masks = sorted(os.listdir("D:/Dataset/Normal All/normal all masks/"))


# In[64]:


normal_images = [cv2.imread(os.path.join("D:/Dataset/Normal All/normal all images/", img)) for img in sorted_normal_images]
normal_masks = [cv2.imread(os.path.join("D:/Dataset/Normal All/normal all masks/", msk)) for msk in sorted_normal_masks]


# In[69]:


files = glob.glob("D:/Dataset/Normal All/normal all images/*.png") 
files = [file.split("images\\")[-1][:-4] for file in files]


# In[72]:


n_images = []
n_masks = []
n_segmented_images = []

n_img_dir = "D:/Dataset/BUSI Segmented/Image/Normal"
n_msk_dir = "D:/Dataset/BUSI Segmented/Mask/Normal"
n_seg_dir = "D:/Dataset/BUSI Segmented/Segmented/Normal"

for i in range(len(normal_images)):
    
    img = normal_images[i]
    img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    n_images.append(img)
    imsave(os.path.join(n_img_dir, files[i] + ".png"), img)
    
    mask = normal_masks[i]
    mask = cv2.resize(mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    n_masks.append(mask)
    imsave(os.path.join(n_msk_dir, files[i] + "_mask.png"), mask)
    
    result = img.copy()
    result[mask == 0] = 0
    result[mask != 0] = img[mask != 0]
    n_segmented_images.append(result)
    imsave(os.path.join(n_seg_dir, files[i] + "_segmented.png"), result)


# In[73]:


for i in range(10):
    image = n_images[i]
    mask = n_masks[i]
    segmented = n_segmented_images[i]
    
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
    fig.savefig(f"D:/Dataset/BUSI Segmented/Example/Normal/{i+1}_segmented_result.png")

