""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2024, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""




import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn, glob
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


# In[16]:


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.summer):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, axs = plt.subplots(1,1,figsize=(8,4))
    img = axs.imshow(cm, interpolation='nearest', cmap=cmap)
    axs.set_title(title)
    plt.colorbar(img)
    tick_marks = np.arange(len(classes))
    axs.set_xticks(tick_marks, classes, rotation=0)
    axs.set_yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    axs.set_ylabel('True label')
    axs.set_xlabel('\nPredicted label')
    plt.tight_layout()
    plt.show()
    
    fig.savefig("/kaggle/working/resnet50v2_confusion_matrix.png")


# In[3]:


train_path = "/kaggle/input/udiat-segmented-mass/UDIAT Segmented Mass/Train"
test_path = "/kaggle/input/udiat-segmented-mass/UDIAT Segmented Mass/Test"


# In[4]:


cm_plot_labels = ["Benign", "Malignant"]
loss_fn = CategoricalCrossentropy(label_smoothing=0.25)
c = 1024

total_train_samples = len(glob.glob(train_path+"/Benign/*.png")) + len(glob.glob(train_path+"/Malignant/*.png")) 
num_train_samples = np.ceil( total_train_samples * 0.9)
num_val_samples = total_train_samples - num_train_samples
num_test_samples = len(glob.glob(test_path+"/Benign/*.png")) + len(glob.glob(test_path+"/Malignant/*.png")) 

train_batch_size = 8
val_batch_size = train_batch_size
test_batch_size = train_batch_size
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
test_steps = np.ceil(num_test_samples / test_batch_size)


# In[5]:


train_batches = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input, 
        validation_split = 0.1).flow_from_directory(
            train_path, 
            target_size = (image_size, image_size),
            batch_size = train_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical", 
            subset = "training")

valid_batches = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input, 
        validation_split = 0.1).flow_from_directory(
            train_path,
            target_size = (image_size, image_size),
            batch_size = val_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical", 
            subset = "validation")

test_batches = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input).flow_from_directory(
            test_path,
            target_size = (image_size, image_size),
            batch_size = test_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical")


# In[6]:


model = tf.keras.applications.resnet_v2.ResNet50V2()
model.trainable = False
for layer in model.layers:
    layer.trainable = False
#     if "BatchNormalization" in layer.__class__.__name__:
#         layer.trainable = True

x = model.output
x = Dense(c, activation = "relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation = "softmax")(x)

model = Model(inputs = model.input, outputs = predictions) 
model.compile(optimizer = Adam(lr=0.01), loss = loss_fn, metrics=["accuracy"])

model.summary()


# In[7]:


# import os
# os.remove("/kaggle/working/resnet50v2_result.png")


# In[8]:


fname = "/kaggle/working/resnet50v2_with_syn_weights.h5"

model_checkpoint = ModelCheckpoint(filepath=fname, monitor="val_accuracy", mode=max, save_best_only=True, verbose=1)
model_earlystop = EarlyStopping(monitor="val_accuracy", mode=max, min_delta=0.001, patience=50, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, mode=max, min_lr=0.00001, patience=5, verbose=1)

history = model.fit_generator(train_batches,
                              steps_per_epoch=train_steps,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=100,
                              verbose=1,
                              callbacks=[model_checkpoint,model_earlystop,reduce_lr])


# In[9]:


model.load_weights("/kaggle/working/resnet50v2_weights.h5")


# In[10]:


test_labels = test_batches.classes

predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)


# In[17]:


cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels)


# In[12]:


print(sklearn.metrics.classification_report(test_labels, predictions.argmax(axis=1), target_names=cm_plot_labels))
print('Accuracy: ', sklearn.metrics.accuracy_score(test_labels, predictions.argmax(axis=1)))
print('F1 Score: ', sklearn.metrics.f1_score(test_labels, predictions.argmax(axis=1), average="macro"))
print('auc score: ',sklearn.metrics.roc_auc_score(test_labels, predictions.argmax(axis=1), multi_class="ovr", average='macro'))


# In[15]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

fig, axs = plt.subplots(1,2,figsize=(8,4))
plt.tight_layout(pad=3.0)

axs[0].plot(acc)
axs[0].plot(val_acc)
axs[0].set_title("Training and Validation Accuracy")
axs[0].set_ylabel("Accuracy")
axs[0].set_xlabel("Epoch")
axs[0].legend(["Training Accuracy", "Validation Accuracy"], loc="lower right")

axs[1].plot(loss)
axs[1].plot(val_loss)
axs[1].set_title("Training and Validation Loss")
axs[1].set_ylabel("Loss")
axs[1].set_xlabel("Epoch")
axs[1].legend(["Training Loss", "Validation Loss"], loc="upper right")

plt.show()
fig.savefig("/kaggle/working/resnet50v2_result.png")


# In[14]:


test_batches.filenames[0:200:20]


# In[13]:


errors = np.where(test_labels != predictions.argmax(axis=1))[0] 
for i in errors:
    print(test_batches.filenames[i])

