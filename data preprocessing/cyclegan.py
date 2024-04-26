""" 
Author : Pallab Chowdhury
Email : chowdhurypall95@gmail.com

Copyright (c) 2024, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""



import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
autotune = tf.data.experimental.AUTOTUNE
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array
import glob
from PIL import Image
from keras.initializers import RandomNormal


# In[ ]:


## Clear output folder

# def remove_folder_contents(folder):
#     for the_file in os.listdir(folder):
#         file_path = os.path.join(folder, the_file)
#         try:
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 remove_folder_contents(file_path)
#                 os.rmdir(file_path)
#         except Exception as e:
#             print(e)

# folder_path = '/kaggle/working/'
# remove_folder_contents(folder_path)
# os.rmdir(folder_path)


# In[ ]:


# # Data Augmentation

# # Set the directory containing the original images
# input_dir = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant'

# # Create Separate Directories for only images
# output_dir1 = "/kaggle/working/malignant"
# # Create Separate Directories for augmented images
# output_dir2 = "/kaggle/working/rotated_malignant"

# # create the output directories if they don't exist
# os.makedirs(output_dir1, exist_ok=True)
# os.makedirs(output_dir2, exist_ok=True)

# # Set the rotation angles (in degrees)
# angles = [90, 180, 270]

# # Loop through the input directory and perform data augmentation
# for i, file_name in enumerate(os.listdir(input_dir)):
    
#     if "_mask" not in file_name:
#         # Load the image
#         img = cv2.imread(os.path.join(input_dir, file_name))

#         # Save the augmented image
#         cv2.imwrite(os.path.join(output_dir1, file_name), img)
        
#         # Loop through the rotation angles and perform data augmentation
#         for angle in angles:
#             # Perform the rotation
#             rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if angle == 90 else cv2.rotate(img, cv2.ROTATE_180) if angle == 180 else cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

#             # Generate the output file name
#             output_file_name = f'{i+1}_{file_name.split(".")[0]}_{angle}.png'

#             # Save the augmented image
#             cv2.imwrite(os.path.join(output_dir2, output_file_name), rotated_img)


# In[3]:


# Data Augmentation

# Set the directory containing the original images
input_dir = '/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/normal'

# Create Separate Directories for only images
output_dir3 = "/kaggle/working/normal"

# create the output directories if they don't exist
os.makedirs(output_dir3, exist_ok=True)




# Loop through the input directory and perform data augmentation
for i, file_name in enumerate(os.listdir(input_dir)):
    
    if "_mask" not in file_name:
        # Load the image
        img = cv2.imread(os.path.join(input_dir, file_name))

        # Save the augmented image
        cv2.imwrite(os.path.join(output_dir3, file_name), img)
        


# In[4]:


# Create Separate Directories for only images
output_dir1 = "/kaggle/working/malignant"
# Create Separate Directories for augmented images
output_dir2 = "/kaggle/working/rotated_malignant"
# Create Separate Directories for normal images
output_dir3 = "/kaggle/working/normal"


# In[5]:


len(os.listdir(output_dir3))
# len(os.listdir(output_dir2))


# In[6]:


# prior = glob.glob("/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant/*_mask.png")
prior = glob.glob("/kaggle/working/normal/*.png")
trn_prior, tst_prior = train_test_split(prior, test_size=0.1, random_state=42)
current = glob.glob("/kaggle/working/malignant/*.png")
trn_current, tst_current = train_test_split(current, test_size=0.1, random_state=42)


# In[7]:


type(trn_prior)


# In[9]:


trn_prior


# In[10]:


# Preprocessing Functions

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label, name):
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    # img = tf.image.resize(img, [*orig_img_size])
    # Random crop to 256X256
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label, name):
    # Only resizing and normalization for the test images.
    #img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img


# In[ ]:


#Create Dataset objects


# In[11]:


size = 256

def generator1():
    for i in range(len(trn_prior)):
        img = Image.open(trn_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        train_x_mydata = np.array(img.resize((size,size), Image.NEAREST))
        train_y_mydata = np.array([0])
        train_name_mydata = np.array([trn_prior[i].split('/')[-1]])
        yield train_x_mydata, train_y_mydata, train_name_mydata
    
def generator2():

    for i in range(len(tst_prior)):
        img = Image.open(tst_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        test_x_mydata = np.array(img.resize((size,size), Image.NEAREST))
        test_y_mydata = np.array([0]) 
        test_name_mydata = np.array([tst_prior[i].split('/')[-1]])
        yield test_x_mydata, test_y_mydata, test_name_mydata
    
def generator3():

    for i in range(len(trn_current)):
        img = Image.open(trn_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        train_x_cbis = np.array(img.resize((size,size), Image.NEAREST))
        train_y_cbis = np.array([1])
        train_name_cbis = np.array([trn_current[i].split('/')[-1]])
        yield train_x_cbis, train_y_cbis, train_name_cbis
    
def generator4():

    for i in range(len(tst_current)):
        img = Image.open(tst_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        test_x_cbis = np.array(img.resize((size,size), Image.NEAREST))
        test_y_cbis = np.array([1])
        test_name_cbis = np.array([tst_current[i].split('/')[-1]])
        yield test_x_cbis, test_y_cbis, test_name_cbis


# In[13]:


train_prior = tf.data.Dataset.from_generator(generator1, (tf.float32, tf.int16, tf.string), 
                                              output_shapes=((size,size,3), (1,), (1,))) 
test_prior = tf.data.Dataset.from_generator(generator2, (tf.float32, tf.int16, tf.string), 
                                             output_shapes=((size,size,3), (1,), (1,)))

train_current = tf.data.Dataset.from_generator(generator3, (tf.float32, tf.int16, tf.string), 
                                                output_shapes=((size,size,3), (1,), (1,)))
test_current = tf.data.Dataset.from_generator(generator4, (tf.float32, tf.int16, tf.string), 
                                               output_shapes=((size,size,3), (1,), (1,)))


# In[14]:


train_prior, train_current, test_prior, test_current


# In[15]:



# Define the standard image size.
orig_img_size = (size, size)
# Size of the random crops to be used during training.
input_img_size = (size, size, 3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed = 42)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed = 42)

buffer_size = size
batch_size = 1


# In[16]:


# Apply the preprocessing operations to the training data
train_prior = (
    train_prior.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))

train_current = (
    train_current.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))

# Apply the preprocessing operations to the test data
test_prior = (
    test_prior.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))

test_current = (
    test_current.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size))


# In[17]:


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.
    Args:
        padding(tuple): Amount of padding for the spatial dimensions.
    Returns:
        A padded tensor with the same type as the input tensor."""

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


# In[18]:


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# In[19]:


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")


# In[20]:


class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity)
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity)

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables))

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables))

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }


# In[ ]:


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test_prior.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Generated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()


# In[21]:


# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# In[22]:


# Create cycle gan model
cycle_gan_model = CycleGan(generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)


# In[23]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_dir = "/kaggle/working/mal_rotated_logs"
os.makedirs(checkpoint_dir, exist_ok=True)


# In[24]:


# Callbacks
checkpoint_filepath = "/kaggle/working/mal_rotated_logs/cyclegan_checkpoints_{epoch:03d}"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, monitor='G_loss',
                             verbose=1,mode='min')
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='G_loss', patience=20)


# In[25]:


cycle_gan_model.fit(
    tf.data.Dataset.zip((train_prior, train_current)),
    epochs=100,
    callbacks=[model_checkpoint_callback,early_stopping_callback]
)


# In[26]:


os.listdir('/kaggle/working/mal_rotated_logs')


# In[28]:


checkpoint_dir = "/kaggle/working/aug_normal2malignant"
os.makedirs(checkpoint_dir, exist_ok=True)


# In[27]:


# Load the checkpoints

weight_file = "/kaggle/working/mal_rotated_logs/cyclegan_checkpoints_049"
cycle_gan_model.load_weights(weight_file).expect_partial()
print("Weights loaded successfully")


# In[29]:


def generator1():
    #train_x_mydata = np.ndarray(shape=(len(trn_mydata), 256, 256, 3),dtype=np.float32)
    #train_y_mydata = np.ndarray(shape=(len(trn_mydata),),dtype=np.int16)
    for i in range(len(trn_prior)):
        img = Image.open(trn_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((size,size), Image.NEAREST)   
        train_x_mydata = (np.asarray(img) / 127.5) - 1.0
        train_y_mydata = np.array([0])
        train_name_mydata = np.array([trn_prior[i].split('\\')[-1]])
        yield train_x_mydata, train_y_mydata, train_name_mydata
    
def generator2():
    #test_x_mydata = np.ndarray(shape=(len(tst_mydata), 256, 256, 3),dtype=np.float32)
    #test_y_mydata = np.ndarray(shape=(len(tst_mydata),),dtype=np.int16)
    for i in range(len(tst_prior)):
        img = Image.open(tst_prior[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((size,size), Image.NEAREST)
        test_x_mydata = (np.asarray(img) / 127.5) - 1.0
        test_y_mydata = np.array([0]) 
        test_name_mydata = np.array([tst_prior[i].split('\\')[-1]])
        yield test_x_mydata, test_y_mydata, test_name_mydata
    
def generator3():
    #train_x_cbis = np.ndarray(shape=(len(trn_cbis), 256, 256, 3),dtype=np.float32)
    #train_y_cbis = np.ndarray(shape=(len(trn_cbis),),dtype=np.int16)
    for i in range(len(trn_current)):
        img = Image.open(trn_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((256,256), Image.NEAREST)
        train_x_cbis = (np.asarray(img) / 127.5) - 1.0
        train_y_cbis = np.array([1])
        train_name_cbis = np.array([trn_current[i].split('\\')[-1]])
        yield train_x_cbis, train_y_cbis, train_name_cbis
    
def generator4():
    #test_x_cbis = np.ndarray(shape=(len(tst_cbis), 256, 256, 3),dtype=np.float32)
    #test_y_cbis = np.ndarray(shape=(len(tst_cbis),),dtype=np.int16)
    for i in range(len(tst_current)):
        img = Image.open(tst_current[i])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((size,size), Image.NEAREST)
        test_x_cbis = (np.asarray(img) / 127.5) - 1.0
        test_y_cbis = np.array([1])
        test_name_cbis = np.array([tst_current[i].split('\\')[-1]])
        yield test_x_cbis, test_y_cbis, test_name_cbis
    
train_prior = tf.data.Dataset.from_generator(generator1, (tf.float32, tf.int16, tf.string))  
test_prior = tf.data.Dataset.from_generator(generator2, (tf.float32, tf.int16, tf.string))   

train_current = tf.data.Dataset.from_generator(generator3, (tf.float32, tf.int16, tf.string))  
test_current = tf.data.Dataset.from_generator(generator4, (tf.float32, tf.int16, tf.string))   

# Define the standard image size.
orig_img_size = (size, size)
# Size of the random crops to be used during training.
input_img_size = (size, size, 3)
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = size
batch_size = 1

# Apply the preprocessing operations to the training data
train_prior = train_prior.cache().shuffle(buffer_size).batch(batch_size)
test_prior = test_prior.cache().shuffle(buffer_size).batch(batch_size)
train_current = train_current.cache().shuffle(buffer_size).batch(batch_size)
test_current = test_current.cache().shuffle(buffer_size).batch(batch_size)


# In[30]:


# save images


for i, (img, _, name) in enumerate(test_prior.take(len(tst_prior))):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    prediction = keras.preprocessing.image.array_to_img(prediction)
    name = name[0].numpy()[0].decode("utf-8")
    prediction.save("/kaggle/working/aug_normal2malignant/"+os.path.basename(name))


# In[31]:


for i, (img, _, name) in enumerate(train_prior.take(len(trn_prior))):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    prediction = keras.preprocessing.image.array_to_img(prediction)
    name = name[0].numpy()[0].decode("utf-8")
    prediction.save("/kaggle/working/aug_normal2malignant/"+os.path.basename(name))


# In[32]:


# !zip -r nor2mal_augmented.zip /kaggle/working/aug_normal2malignant/

