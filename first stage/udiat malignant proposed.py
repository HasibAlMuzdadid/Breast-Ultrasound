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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Layer, Concatenate, Conv2D, BatchNormalization, ReLU, UpSampling2D, AveragePooling2D
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# from data import load_train_data, load_test_data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
from tensorflow.keras.applications import ResNet50


# In[2]:


img_rows = 256
img_cols = 256
smooth = 1.


# In[3]:


def dice_coef(y_true, y_pred):
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


class ConvBlock(Layer):
    
    def __init__(self, filters=256, kernel_size=3, use_bias=False, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        
        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', dilation_rate=dilation_rate, use_bias=use_bias, kernel_initializer='he_normal'),
            BatchNormalization(),
            ReLU()
        ])
    
    def call(self, X): return self.net(X)        
        
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "kernel_size":self.kernel_size,
            "use_bias":self.use_bias,
            "dilation_rate":self.dilation_rate
        }


# In[5]:


def AtrousSpatialPyramidPooling(X):
    
    # Shapes 
    _, height, width, _ = X.shape
    
    # Image Pooling 
    image_pool = AveragePooling2D(pool_size=(height, width), name="ASPP-AvgPool2D")(X)
    image_pool = ConvBlock(kernel_size=1, name="ASPP-ConvBlock-1")(image_pool)
    image_pool = UpSampling2D(size=(height//image_pool.shape[1], width//image_pool.shape[2]), name="ASPP-UpSampling")(image_pool)
    
    # Conv Blocks
    conv_1 = ConvBlock(kernel_size=1, dilation_rate=1, name="ASPP-Conv-1")(X)
    conv_6 = ConvBlock(kernel_size=3, dilation_rate=6, name="ASPP-Conv-6")(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12, name="ASPP-Conv-12")(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18, name="ASPP-Conv-18")(X)
    
    # Concat All
    concat = Concatenate(axis=-1, name="ASPP-Concat")([image_pool, conv_1, conv_6, conv_12, conv_18])
    net = ConvBlock(kernel_size=1, name="ASPP-Net")(concat)
    
    return net


# In[6]:


def get_model():

    # Input
    InputL = Input(shape=(img_cols, img_rows, 3), name="InputLayer")

    # Base Mode
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=InputL)

    # ASPP Phase
    DCNN = resnet50.get_layer('conv4_block6_2_relu').output
    ASPP = AtrousSpatialPyramidPooling(DCNN)
    ASPP = UpSampling2D(size=(img_cols//4//ASPP.shape[1], img_rows//4//ASPP.shape[2]), name="AtrousSpatial")(ASPP)

    # LLF Phase
    LLF = resnet50.get_layer('conv2_block3_2_relu').output
    LLF = ConvBlock(filters=48, kernel_size=1, name="LLF-ConvBlock")(LLF)

    # Combined
    combined = Concatenate(axis=-1, name="Combine-LLF-ASPP")([ASPP, LLF])
    features = ConvBlock(name="Top-ConvBlock-1")(combined)
    features = ConvBlock(name="Top-ConvBlock-2")(features)
    upsample = UpSampling2D(size=(img_cols//features.shape[1], img_rows//features.shape[1]), interpolation='bilinear', name="Top-UpSample")(features)

    # Output Mask
    PredMask = Conv2D(1, kernel_size=(1,1), strides=1, padding='same', activation='sigmoid', use_bias=False, name="OutputMask")(upsample)

    # Model
    model = Model(inputs=[InputL], outputs=[PredMask], name="Proposed-Model")
    model.compile(optimizer=Adam(1e-3), loss=[dice_coef_loss], metrics=[dice_coef, iou_coef, "accuracy"])

    return model


# In[7]:


m = get_model()
m.summary()


# In[8]:


# name = "malignant"
# imgs_train, imgs_mask_train = load_train_data("malignant")

imgs_train = np.load("/kaggle/input/udiat-malignant/imgs_train_malignant.npy")
imgs_mask_train = np.load("/kaggle/input/udiat-malignant/imgs_mask_train_malignant.npy")
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
    fig.savefig(f"/kaggle/working/malignant_fold_{fold+1}_model_evaluation.png")


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
    
    model = get_model()

    fname = "/kaggle/working/malignant_model_fold_0"+str(fold+1)+"_weights.h5"

    model_checkpoint = ModelCheckpoint(filepath=fname, monitor="val_loss",mode=min, save_best_only=True,verbose=1)
    model_earlystop = EarlyStopping(monitor="val_loss",mode=min,min_delta=0.001,patience=50,verbose=1)
    
    history = model.fit(train_generator,
                    batch_size=8, epochs=200, verbose=1, shuffle=False,
                    validation_data=val_generator,steps_per_epoch=len(train)/8,
                    validation_steps=len(val)/8,
                    callbacks=[model_checkpoint,model_earlystop])
    
    visuals(history)


# # Checking Test Set First
# In[11]:


# imgs_test, imgs_id_test = load_test_data("malignant")

imgs_test = np.load("/kaggle/input/udiat-malignant/imgs_test_malignant.npy")
imgs_id_test = np.load("/kaggle/input/udiat-malignant/imgs_id_test_malignant.npy")
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


model1 = get_model()
model1.load_weights("/kaggle/input/udiat-malignant-model-3-fold-weights/malignant_model_fold_01_weights.h5")
model2 = get_model()
model2.load_weights("/kaggle/input/udiat-malignant-model-3-fold-weights/malignant_model_fold_02_weights.h5")
model3 = get_model()
model3.load_weights("/kaggle/input/udiat-malignant-model-3-fold-weights/malignant_model_fold_03_weights.h5")

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


# imgs_mask_test = img_3

imgs_mask_test = ((0.4*img_1) + (0.2*img_2) + (0.4*img_3)) / (0.4 + 0.2 + 0.4)


# imgs_mask_test = ((0.3*img_1) + (0.3*img_2) + (0.3*img_3)) / (0.3 + 0.3 + 0.3)
# imgs_mask_test = (img_1 + img_2 + img_3) / (1 + 1 + 1)
# imgs_mask_test = ((1.0*img_1) + (1.0*img_2) + (0.0*img_3)) / (1.0 + 1.0 + 0.0)  # image 0.1
# imgs_mask_test = ((0.9*imgs_mask_test_1) + (0.5*imgs_mask_test_2) + (0.8*imgs_mask_test_3)) / (0.9 + 0.5 + 0.8)


# In[16]:


imgs_mask_test[imgs_mask_test > 0.4] = 1.0
imgs_mask_test[imgs_mask_test <= 0.4] = 0.0
# imgs_mask_test.min()


# In[17]:


imgs_id_t = imgs_id_test
imgs_id_t = imgs_id_t.astype('float32')
imgs_id_t = imgs_id_t[..., np.newaxis]
imgs_id_t = imgs_id_t // 255

dice = dice_coef(imgs_id_t, imgs_mask_test)
print(f"Dice Score : {dice*100}")

iou = iou_coef(imgs_id_t, imgs_mask_test)
print(f"IOU Score : {iou*100}")


# In[ ]:


np.save(os.path.join("/kaggle/working", "malignant_imgs_mask_test_model.npy"), imgs_mask_test)
print("Saving to .npy files done")


# In[ ]:


import shutil
shutil.rmtree("/kaggle/working/test_result")


# In[20]:


# os.makedirs("/kaggle/working/test_result", exist_ok=True)

for i in range(30):
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
    #fig.savefig(f"/kaggle/working/test_result/{i+1}_malignant_model_test_result.png")


# # Checking Generated Images and Predicting Masks
# In[ ]:


# loading and preprocessing generated images
data_path = "/kaggle/input/udiat-malignant-generated-image/CycleGAN"  
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


np.save(os.path.join("/kaggle/working", "malignant_generated_images.npy"), imgs)
print("Saving to .npy files done")


# In[ ]:


generated_imgs = np.load("/kaggle/working/malignant_generated_images.npy")
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


gen_imgs_mask = ((0.4*gen_imgs_mask_1) + (0.2*gen_imgs_mask_2) + (0.4*gen_imgs_mask_3)) / (0.4 + 0.2 + 0.4)


# In[ ]:


gen_imgs_mask[gen_imgs_mask > 0.4] = 1.0
gen_imgs_mask[gen_imgs_mask <= 0.4] = 0.0

gen_imgs_mask.min()


# In[ ]:


# import shutil
# shutil.rmtree("/kaggle/working/malignant_generated_masks")
# os.remove("/kaggle/working/generated_masks.zip")

pred_dir = "/kaggle/working/udiat_malignant_generated_masks"

# create the output directories if they don't exist
os.makedirs(pred_dir, exist_ok=True)


# In[ ]:


files = glob.glob("/kaggle/input/udiat-malignant-generated-image/CycleGAN/*.png") 
len(files)
files[0]


# In[ ]:


files = [file.split('CycleGAN/')[-1][:-4] for file in files]
idx = 0
for image in gen_imgs_mask:
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, files[idx] + '_mask.png'), image)
    idx = idx + 1


# In[ ]:


# Sort the list of filenames
# filenames = sorted(os.listdir(folder_path))
sorted_gen_images = sorted(os.listdir("/kaggle/input/udiat-malignant-generated-image/CycleGAN/"))
sorted_gen_masks = sorted(os.listdir("/kaggle/working/udiat_malignant_generated_masks/"))

# Load the images and masks from the folder
gen_images = [cv2.imread(os.path.join("/kaggle/input/udiat-malignant-generated-image/CycleGAN/", img)) for img in sorted_gen_images]
gen_masks = [cv2.imread(os.path.join("/kaggle/working/udiat_malignant_generated_masks/", msk)) for msk in sorted_gen_masks]

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
    fig.savefig(f"/kaggle/working/{i+1}_model_result.png")


# In[ ]:


# !zip -r generated_masks.zip /kaggle/working/udiat_malignant_generated_masks/

