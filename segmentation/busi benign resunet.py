""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2024, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""



import os, glob
from skimage.io import imsave
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Add, MaxPooling2D, Activation, Dense, Reshape, GlobalAveragePooling2D, Multiply, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# from data import load_train_data, load_test_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# In[2]:


img_rows = 256
img_cols = 256
smooth = 1


# In[3]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred)+0.6*iou_coef(y_true, y_pred))

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# In[4]:


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


# In[5]:


def resnet_block(x, n_filter, strides=1):
    x_init = x
    
    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)
    
    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


# In[6]:


def get_resunet():    
    inputs = Input((img_rows, img_cols, 3))
    
    conv1 = resnet_block(inputs,32 , strides=1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = resnet_block(pool1,64 , strides=1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = resnet_block(pool2, 128, strides=1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = resnet_block(pool3, 256, strides=1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = resnet_block(up6, 256, strides=1)
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = resnet_block(up7, 128, strides=1)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = resnet_block(up8, 64, strides=1)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = resnet_block(up9, 32, strides=1)
            
    conv9 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv9])
    model.compile(optimizer=Adam(1e-3), loss=[dice_coef_loss], metrics=[dice_coef, iou_coef, "accuracy"])
    return model


# In[7]:


m = get_resunet()
m.summary()


# In[8]:


imgs_train = np.load("/kaggle/input/busi-benign/BUSI Benign Image Mask/imgs_train_benign.npy")
imgs_mask_train = np.load("/kaggle/input/busi-benign/BUSI Benign Image Mask/imgs_mask_train_benign.npy")
imgs_train.shape, imgs_mask_train.shape


# In[9]:


imgs_train = imgs_train.astype('float32')

mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

imgs_mask_train = imgs_mask_train.astype('float32')

imgs_mask_train /= 255.  # scale masks to [0, 1]
imgs_mask_train = imgs_mask_train[..., np.newaxis]
imgs_train.shape, imgs_mask_train.shape


# In[10]:


data_gen_args = dict(rotation_range=40,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     shear_range=0.1,
                     horizontal_flip=True,
                     fill_mode='reflect')
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(imgs_train, augment=True, seed=seed)
mask_datagen.fit(imgs_mask_train, augment=True, seed=seed)


# In[ ]:


def visuals(history):
    
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    plt.tight_layout(pad=3.0)

    axs[0,0].plot(history.history['dice_coef'])
    axs[0,0].plot(history.history['val_dice_coef'])
    axs[0,0].set_title('model dice coef')
    axs[0,0].set_ylabel('dice coef')
    axs[0,0].set_xlabel('epoch')
    axs[0,0].legend(['train', 'test'], loc='upper left')

    axs[0,1].plot(history.history['iou_coef'])
    axs[0,1].plot(history.history['val_iou_coef'])
    axs[0,1].set_title('model iou coef')
    axs[0,1].set_ylabel('iou coef')
    axs[0,1].set_xlabel('epoch')
    axs[0,1].legend(['train', 'test'], loc='upper left')

    axs[1,0].plot(history.history['loss'])
    axs[1,0].plot(history.history['val_loss'])
    axs[1,0].set_title('model loss')
    axs[1,0].set_ylabel('loss')
    axs[1,0].set_xlabel('epoch')
    axs[1,0].legend(['train', 'test'], loc='upper left')
    
    axs[1,1].plot(history.history['accuracy'])
    axs[1,1].plot(history.history['val_accuracy'])
    axs[1,1].set_title('model accuracy')
    axs[1,1].set_ylabel('accuracy')
    axs[1,1].set_xlabel('epoch')
    axs[1,1].legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig(f"/kaggle/working/benign_fold_{fold+1}_resunet_evaluation.png")


# In[ ]:


for fold, (train_index,val_index) in enumerate(KFold(n_splits = 3, shuffle=False).split(imgs_train,imgs_mask_train)):
    
    tf.keras.backend.clear_session()
    print(f"Fold : {fold+1}")
    train = imgs_train[train_index]
    train_mask = imgs_mask_train[train_index]
    
    val = imgs_train[val_index]
    val_mask = imgs_mask_train[val_index]
    
    train_image_generator = image_datagen.flow(train,  seed=seed)
    train_mask_generator = mask_datagen.flow(train_mask,  seed=seed)
    # combine generators into one which yields image and masks
    train_generator = zip(train_image_generator, train_mask_generator)
    
    val_image_generator = image_datagen.flow(val,  seed=seed)
    val_mask_generator = mask_datagen.flow(val_mask,  seed=seed)
    # combine generators into one which yields image and masks
    val_generator = zip(val_image_generator, val_mask_generator)
    
    model = get_resunet()

    fname = "/kaggle/working/benign_resunet_fold_0"+str(fold+1)+"_weights.h5"

    model_checkpoint = ModelCheckpoint(filepath=fname, monitor="val_loss",mode=max, save_best_only=True,verbose=1)
    model_earlystop = EarlyStopping(monitor="val_loss",mode=min,min_delta=0.001,patience=50,verbose=1)
    
    history = model.fit(train_generator,
                    batch_size=32, epochs=200, verbose=1, shuffle=False,
                    validation_data=val_generator,steps_per_epoch=len(train)/32,
                    validation_steps=len(val)/32,
                    callbacks=[model_checkpoint,model_earlystop])
    
    visuals(history)


# # Checking Test Set First
# In[11]:


# imgs_test, imgs_id_test = load_test_data("benign")

imgs_test = np.load("/kaggle/input/busi-benign/BUSI Benign Image Mask/imgs_test_benign.npy")
imgs_id_test = np.load("/kaggle/input/busi-benign/BUSI Benign Image Mask/imgs_id_test_benign.npy")
imgs_test.shape, imgs_id_test.shape


# In[12]:


imgs_t = imgs_test
imgs_t = imgs_t.astype('float32')

mean = np.mean(imgs_t)  # mean for data centering
std = np.std(imgs_t)  # std for data normalization

imgs_t -= mean
imgs_t /= std

imgs_t.shape


# In[13]:


model1 = get_resunet()
model1.load_weights("/kaggle/input/benign-resunet-3-fold-weights/benign resunet 3 fold weights/benign_resunet_fold_01_weights.h5")
model2 = get_resunet()
model2.load_weights("/kaggle/input/benign-resunet-3-fold-weights/benign resunet 3 fold weights/benign_resunet_fold_02_weights.h5")
model3 = get_resunet()
model3.load_weights("/kaggle/input/benign-resunet-3-fold-weights/benign resunet 3 fold weights/benign_resunet_fold_03_weights.h5")

imgs_mask_test_1 = model1.predict(imgs_t, verbose=1)
print("Fold 1 done")
imgs_mask_test_2 = model2.predict(imgs_t, verbose=1)
print("Fold 2 done")
imgs_mask_test_3 = model3.predict(imgs_t, verbose=1)
print("Fold 3 done")


# In[14]:


img_1 = imgs_mask_test_1
img_2 = imgs_mask_test_2
img_3 = imgs_mask_test_3


# In[15]:


img_1.min(),img_1.max()


# In[16]:


# imgs_mask_test = ((0.8*img_1) + (0.1*img_2) + (0.8*img_3)) / (0.8 + 0.1 + 0.8)
imgs_mask_test = ((0.4*img_1) + (0.2*img_2) + (0.4*img_3)) / (0.4 + 0.2 + 0.4)
# imgs_mask_test = (img_1 + img_2 + img_3) / (1 + 1 + 1)
# imgs_mask_test = ((0.9*imgs_mask_test_1) + (0.0*imgs_mask_test_2) + (0.1*imgs_mask_test_3)) / (0.9 + 0.0 + 0.1)


# In[17]:


imgs_mask_test[imgs_mask_test < 0.25] = 0.0
imgs_mask_test[imgs_mask_test >= 0.25] = 1.0


# In[18]:


imgs_id_t = imgs_id_test
imgs_id_t = imgs_id_t.astype('float32')
imgs_id_t = imgs_id_t[..., np.newaxis]
imgs_id_t = imgs_id_t // 255

dice = dice_coef(imgs_id_t, imgs_mask_test)
print(f"Dice Score : {dice*100}")

iou = iou_coef(imgs_id_t, imgs_mask_test)
print(f"IOU Score : {iou*100}")


# In[ ]:


np.save(os.path.join("/kaggle/working", "benign_imgs_mask_test_resunet.npy"), imgs_mask_test)
print("Saving to .npy files done")


# In[21]:


# os.makedirs("/kaggle/working/test_result", exist_ok=True)

for i in range(16):
    # Select an image and its true mask
    test_image = imgs_test[i]
    test_mask = imgs_id_test[i]
    pred_mask = imgs_mask_test[i]
    #imsave(os.path.join("/kaggle/working", f"{i+1}_pred_mask.png"), pred_mask)
    cv2.imwrite(f"{i+1}_pred_mask.png",pred_mask*255)

    test_image = cv2.resize(test_image, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    pred_mask = cv2.resize(pred_mask, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)   
    seg_mask = test_image.copy()
    seg_mask[pred_mask == 0] = 0
    seg_mask[pred_mask != 0] = test_image[pred_mask != 0]
    #imsave(os.path.join("/kaggle/working", f"{i+1}_Segmented_mask.png"), seg_mask)



    # Display the image and the true mask
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(test_image, cmap="gray")
    ax1.set_title("Image")
    ax1.set_axis_off()
    ax2.imshow(test_mask, cmap="gray")
    ax2.set_title("True Mask")
    ax2.set_axis_off()
    ax3.imshow(pred_mask, cmap="gray")
    ax3.set_title("Predicted Mask")
    ax3.set_axis_off()
    ax4.imshow(seg_mask, cmap="gray")
    ax4.set_title("Segmented")
    ax4.set_axis_off()

    plt.show()
    #fig.savefig(f"/kaggle/working/test_result/{i+1}_benign_resunet_test_result.png")


# # Checking Generated Images and Predicting Masks
# In[ ]:


# loading and preprocessing generated images
data_path = "/kaggle/input/benign-generated-images"  
images = os.listdir(data_path)
total = len(images)
imgs = np.ndarray((total, img_rows, img_cols, 3), dtype=np.uint8)

i = 0
print(f"{'-'*30} creating generated images {'-'*30}")
for j in range(len(images)):

    img = cv2.imread(os.path.join(data_path, images[j]))
    #img = cv2.resize(img, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
    #enhancement
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img_eqhist=cv2.equalizeHist(gray_img)
    img = cv2.cvtColor(gray_img_eqhist, cv2.COLOR_GRAY2BGR)
    #end enhancement
    img = np.array([img])
    imgs[i] = img
    if i % 100 == 0:
        print(f"Done: {i}/{total} images")
    i += 1
print("Loading done")


np.save(os.path.join("/kaggle/working", "benign_generated_images.npy"), imgs)
print("Saving to .npy files done")


# In[ ]:


generated_imgs = np.load("/kaggle/working/benign_generated_images.npy")
generated_imgs.shape


# In[ ]:


generated_imgs = generated_imgs.astype('float32')

mean = np.mean(generated_imgs)  # mean for data centering
std = np.std(generated_imgs)  # std for data normalization

generated_imgs -= mean
generated_imgs /= std

generated_imgs.shape


# In[ ]:


gen_imgs_mask_1 = model1.predict(generated_imgs, verbose=1)
print("Fold 1 done")
gen_imgs_mask_2 = model2.predict(generated_imgs, verbose=1)
print("Fold 2 done")
gen_imgs_mask_3 = model3.predict(generated_imgs, verbose=1)
print("Fold 3 done")


# In[ ]:


gen_imgs_mask = ((0.9*gen_imgs_mask_1) + (0.0*gen_imgs_mask_2) + (0.1*gen_imgs_mask_3)) / (0.9 + 0.0 + 0.1)


# In[ ]:


gen_imgs_mask[gen_imgs_mask < 0.6] = 0.0
gen_imgs_mask[gen_imgs_mask >= 0.6] = 1.0
gen_imgs_mask.min()


# In[ ]:


# import shutil
# shutil.rmtree("/kaggle/working/benign_generated_masks")
# os.remove("/kaggle/working/generated_masks.zip")

pred_dir = "/kaggle/working/benign_generated_masks"
os.makedirs(pred_dir, exist_ok=True)


# In[ ]:


files = glob.glob("/kaggle/input/benign-generated-images/*.png") 
len(files)
# files[0]


# In[ ]:


files = [file.split('images/')[-1][:-4] for file in files]
idx = 0
for image in gen_imgs_mask:
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, files[idx] + '_mask.png'), image)
    idx = idx + 1


# In[ ]:


# Sort the list of filenames
# filenames = sorted(os.listdir(folder_path))
sorted_gen_images = sorted(os.listdir("/kaggle/input/benign-generated-images/"))
sorted_gen_masks = sorted(os.listdir("/kaggle/working/benign_generated_masks/"))

# Load the images and masks from the folder
gen_images = [cv2.imread(os.path.join("/kaggle/input/benign-generated-images/", img)) for img in sorted_gen_images]
gen_masks = [cv2.imread(os.path.join("/kaggle/working/benign_generated_masks/", msk)) for msk in sorted_gen_masks]

# Convert the images and masks to numpy arrays
images = np.array(gen_images)
masks = np.array(gen_masks)

# images[300],masks[300]


# In[ ]:


for i in range(5):
    image = images[i]
    mask = masks[i]
    
    # Display the image and the true mask
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_axis_off()
    ax1.set_title("Generated Image")
    ax2.imshow(mask)
    ax2.set_axis_off()
    ax2.set_title("Generated Mask")

    plt.show()
    fig.savefig(f"/kaggle/working/{i+1}_resunet_generated_image_result.png")


# In[ ]:


# !zip -r generated_masks.zip /kaggle/working/benign_generated_masks/

