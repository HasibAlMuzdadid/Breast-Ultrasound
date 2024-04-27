""" 
Author : Md Hasib Al Muzdadid Haque Himel
Email : muzdadid@gmail.com

Copyright (c) 2024, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""



# !pip install --upgrade tensorflow==2.9.0


# In[2]:


import tensorflow as tf
tf.__version__


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import itertools
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn, glob
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


# In[4]:


import seaborn as sns
palette = ["#11264e","#00507A","#026e90","#008b99","#6faea4","#fcdcb0","#FEE08B","#faa96e","#f36b3b","#ef3f28","#CC0028"]
palette_cmap=["#CC0028","#ef3f28","#f36b3b","#faa96e","#FEE08B","#fcdcb0","#6faea4","#008b99","#026e90","#00507A","#11264e"]

sns.palplot(sns.color_palette(palette))
sns.palplot(sns.color_palette(palette_cmap))
plt.show()


# In[5]:


cmap = col.LinearSegmentedColormap.from_list("", ["#d64c5e", "#d8d0b4"])
# cmap = col.LinearSegmentedColormap.from_list("", ["#1F9A82","#e5e5e5"])
# cmap = col.LinearSegmentedColormap.from_list("", ["#b85550","#FEE08B"])
# cmap = col.LinearSegmentedColormap.from_list("", ["#e1d5e7","#ffcc99"])
cmap


# In[47]:


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap = cmap):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, axs = plt.subplots(1,1,figsize=(4,4))
    img = axs.imshow(cm, interpolation='nearest', cmap=cmap, alpha=1.0)
    # axs.set_title(title)
    im_ratio = cm.shape[0]/cm.shape[1]
    
    plt.colorbar(img,fraction=0.045*im_ratio).outline.set_visible(False)
    tick_marks = np.arange(len(classes))
    axs.set_xticks(tick_marks, classes, rotation=0, fontweight="normal", fontsize=10)
    axs.set_yticks(tick_marks, classes, rotation=0, fontweight="normal", fontsize=10)
    #axs.grid(which="minor", color="#11264e", linestyle='-', linewidth=4)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),   #for percent
        #plt.text(j, i, format(cm[i, j]),  # for number       
                 horizontalalignment="center",
                 color="#d64c5e" if cm[i, j] > thresh else "#d8d0b4",
                 fontsize=11)

    axs.set_ylabel('True label', weight="normal",fontsize=11)
    axs.set_xlabel('Predicted label', weight="normal",fontsize=11)
    sns.despine(left=True, right=True, top=True, bottom=True)
    plt.tight_layout()
    plt.show()
    
    #fig.savefig("/kaggle/working/stacked_ensemble_confusion_matrix.png", transparent=True, dpi=300,bbox_inches='tight',pad_inches=0.2)


# In[7]:


train_path = "/kaggle/input/thammasat-segmented-mass/Thammasat Segmented Mass v2/Thammasat Segmented Mass/Train"
test_path = "/kaggle/input/thammasat-segmented-mass/Thammasat Segmented Mass v2/Thammasat Segmented Mass/Test"


# In[8]:


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


# In[9]:


train_batches_e = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.efficientnet.preprocess_input, 
        validation_split = 0.1,
        rotation_range = 40,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.2,
        shear_range = 0.1,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'reflect').flow_from_directory(
            train_path, 
            target_size = (image_size, image_size),
            batch_size = train_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical", 
            subset = "training")

valid_batches_e = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.efficientnet.preprocess_input, 
        validation_split = 0.1).flow_from_directory(
            train_path,
            target_size = (image_size, image_size),
            batch_size = val_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical", 
            subset = "validation")

test_batches_e = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
            test_path,
            target_size = (image_size, image_size),
            batch_size = test_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical")


# In[10]:


train_batches_n = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.nasnet.preprocess_input, 
        validation_split = 0.1,
        rotation_range = 40,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.2,
        shear_range = 0.1,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'reflect').flow_from_directory(
            train_path, 
            target_size = (image_size, image_size),
            batch_size = train_batch_size,
            shuffle = False,
            seed = 2,
            class_mode = "categorical", 
            subset = "training")

valid_batches_n = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.nasnet.preprocess_input, 
        validation_split = 0.1).flow_from_directory(
            train_path,
            target_size = (image_size, image_size),
            batch_size = val_batch_size,
            shuffle = False,
            seed = 2,
            class_mode = "categorical", 
            subset = "validation")

test_batches_n = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.nasnet.preprocess_input).flow_from_directory(
            test_path,
            target_size = (image_size, image_size),
            batch_size = test_batch_size,
            shuffle = False,
            seed = 2,
            class_mode = "categorical")


# In[11]:


train_batches_r = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input, 
        validation_split = 0.1,
        rotation_range = 40,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.2,
        shear_range = 0.1,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'reflect').flow_from_directory(
            train_path, 
            target_size = (image_size, image_size),
            batch_size = train_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical", 
            subset = "training")

valid_batches_r = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input, 
        validation_split = 0.1).flow_from_directory(
            train_path,
            target_size = (image_size, image_size),
            batch_size = val_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical", 
            subset = "validation")

test_batches_r = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input).flow_from_directory(
            test_path,
            target_size = (image_size, image_size),
            batch_size = test_batch_size,
            shuffle = False,
            seed = 42,
            class_mode = "categorical")


# In[12]:


model_e = tf.keras.applications.efficientnet.EfficientNetB7()
model_n = tf.keras.applications.nasnet.NASNetLarge()
model_r = tf.keras.applications.resnet_v2.ResNet50V2()

models = [model_e, model_n, model_r]

for i in range(len(models)):
    x = models[i].output
    x = Dense(c, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(2, activation='softmax')(x)
    models[i] = Model(inputs=models[i].input, outputs=predictions) 


# In[13]:


model_e = models[0]        
model_e.load_weights("/kaggle/input/thammasat-weights/Weights v2/efficientnetb7_weights.h5")

model_n = models[1]       
model_n.load_weights("/kaggle/input/thammasat-weights/Weights v2/nasnetlarge_weights.h5")

model_r = models[2]       
model_r.load_weights("/kaggle/input/thammasat-weights/Weights v2/resnet50v2_weights.h5")


# In[14]:


predictions_e = model_e.predict_generator(test_batches_e, steps=test_steps, verbose=1)
np.save("/kaggle/working/pred_efficientnetb7.npy", predictions_e)


# In[15]:


predictions_n = model_n.predict_generator(test_batches_n, steps=test_steps, verbose=1)
np.save("/kaggle/working/pred_nasnetlarge.npy", predictions_n)


# In[16]:


predictions_r = model_r.predict_generator(test_batches_r, steps=test_steps, verbose=1)
np.save("/kaggle/working/pred_resnet50v2.npy", predictions_r)


# In[17]:


train_labels_e = train_batches_e.classes
train_labels = to_categorical(train_labels_e)

valid_labels_e = valid_batches_e.classes
valid_labels = to_categorical(valid_labels_e)

test_labels_e = test_batches_e.classes
test_labels = to_categorical(test_labels_e)


# In[20]:


model_r.get_layer("drop")


# In[21]:


model_e = Model(inputs=model_e.input, outputs=model_e.get_layer("dropout").output)
model_n = Model(inputs=model_n.input, outputs=model_n.get_layer("dropout_1").output)
model_r = Model(inputs=model_r.input, outputs=model_r.get_layer("dropout_2").output)


# In[22]:


pred_train_e = model_e.predict_generator(train_batches_e, steps=train_steps, verbose=1)
pred_train_n = model_n.predict_generator(train_batches_n, steps=train_steps, verbose=1)
pred_train_r = model_r.predict_generator(train_batches_r, steps=train_steps, verbose=1)


# In[23]:


pred_valid_e = model_e.predict_generator(valid_batches_e, steps=val_steps, verbose=1)
pred_valid_n = model_n.predict_generator(valid_batches_n, steps=val_steps, verbose=1)
pred_valid_r = model_r.predict_generator(valid_batches_r, steps=val_steps, verbose=1)


# In[24]:


pred_test_e = model_e.predict_generator(test_batches_e, steps=test_steps, verbose=1)
pred_test_n = model_n.predict_generator(test_batches_n, steps=test_steps, verbose=1)
pred_test_r = model_r.predict_generator(test_batches_r, steps=test_steps, verbose=1)


# In[25]:


def define_stacked_model(models):
    
    for i, model in enumerate(models):
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
            layer._name = "ensemble_" + str(i+1) + "_" + layer.name
    
    ensemble_outputs = [model.output for model in models]

    merge = concatenate(ensemble_outputs)
    hidden = Dense(1000, activation="sigmoid")(merge)   
    hidden2 = Dense(100, activation="relu")(hidden)
    hidden3 = Dense(10, activation="sigmoid")(hidden2)    
    output = Dense(2, activation="softmax")(hidden3)
    
    model = Model(inputs=ensemble_outputs, outputs=output)
    model.compile(optimizer = Adam(lr=0.01), loss = loss_fn, metrics=["accuracy"])
    
    return model


# In[26]:


models = [model_e, model_n, model_r]

stacked_model = define_stacked_model(models)


# In[27]:


stacked_model.summary()


# In[28]:


plot_model(stacked_model)


# In[29]:


train_batches = [pred_train_e, pred_train_n, pred_train_r]
valid_batches = [pred_valid_e, pred_valid_n, pred_valid_r]
test_batches = [pred_test_e, pred_test_n, pred_test_r]


# In[33]:


# train_batches


# In[44]:


fname = "/kaggle/working/stacked_ensemble_weights.h5"

model_checkpoint = ModelCheckpoint(filepath=fname, monitor="accuracy", mode=max, save_best_only=True, verbose=1)
model_earlystop = EarlyStopping(monitor="accuracy", mode=max, min_delta=0.001, patience=50, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="accuracy", factor=0.5, mode=max, min_lr=0.00001, patience=5, verbose=1)


history = stacked_model.fit(train_batches,train_labels,
                            steps_per_epoch=train_steps,
                            validation_data=(valid_batches,valid_labels),
                            validation_steps=val_steps,
                            epochs=200,
                            verbose=1,
                            callbacks=[model_checkpoint,model_earlystop,reduce_lr])


# In[51]:


stacked_model.load_weights("/kaggle/working/stacked_ensemble_weights.h5")


# In[52]:


predictions = stacked_model.predict(test_batches, steps=test_steps, verbose=1)


# In[55]:


np.save("/kaggle/working/pred_stacked_ensemble.npy", predictions)


# In[53]:


cm = confusion_matrix(test_labels_e, predictions.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels)


# In[54]:


print(sklearn.metrics.classification_report(test_labels_e, predictions.argmax(axis=1), target_names=cm_plot_labels))
print('Accuracy: ', sklearn.metrics.accuracy_score(test_labels_e, predictions.argmax(axis=1)))
print('F1 Score: ', sklearn.metrics.f1_score(test_labels_e, predictions.argmax(axis=1), average="macro"))
print('auc score: ',sklearn.metrics.roc_auc_score(test_labels_e, predictions.argmax(axis=1), multi_class="ovr", average='macro'))


# In[41]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

fig, axs = plt.subplots(1,2,figsize=(8,3))
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
fig.savefig("/kaggle/working/stacked_ensemble_result.png")

