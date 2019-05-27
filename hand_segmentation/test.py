# TensorFlow and tf.keras
import tensorflow as tf

# import keras
from keras.layers import *
from keras.models import Model
from keras import losses, models
# Helper libraries

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Images

import imageio
from matplotlib import pyplot as plt

# Paths
from os import path as op


# NETWORK COMPONENTS

# Convolution block
def conv_block(input_tensor, num_filters, kernel_size=3):
    # First layer
    encoder = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same')(input_tensor)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    # Second layer
    encoder = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)

    return encoder


# Encoder block
def encoder_block(input_tensor, num_filters, pool_size=2, strides=2):
    # Encoder
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(strides, strides))(encoder)

    return encoder_pool, encoder


# Decoder block
def decoder_block(input_tensor, concat_tensor, num_filters, kernel_size=3, transpose_kernel_size=2, strides=2):
    # Conv2DTranspose AKA Deconvolution
    decoder = Conv2DTranspose(filters=num_filters, kernel_size=(transpose_kernel_size, transpose_kernel_size),
                              strides=(strides, strides), padding='same')(input_tensor)
    # Concatenate
    decoder = concatenate([concat_tensor, decoder], axis=-1)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    return decoder


# Custom metrics
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
# ----------------------------------------------------------------------------------------------------------------------

# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# mini
mini_root_path = op.join(data_root_path, 'mini')
mini_all_features_path = op.join(mini_root_path, 'all_features')
mini_target_path = op.join(mini_root_path, 'target')
# output model
save_model_dir = op.join(project_root_path, 'models')
save_model_path = op.join(save_model_dir, 'first_try.hdf5')

# Get data
print("Getting data")
mini_train = sio.loadmat(op.join(mini_all_features_path, 'features_train.mat')).get('features_train')
mini_target = sio.loadmat(op.join(mini_target_path, 'target_train.mat')).get('target_train')

# target from uint8 to float64
mini_target = mini_target*0.1
# mini_train from dtype to float64
mini_train = mini_train.astype(float)

# Transpose to get NxMx4
print("Transposing tensors")
mini_train = mini_train.transpose(3, 0, 1, 2)  # 126x200x200x4
mini_target = mini_target.transpose(3, 0, 1, 2)
# TODO images should be shuffled and rotated

# Some parameters
sub_size = 32  # Number of training inputs
batch_size = 2
epochs = 5

# DEBUG
mini_train = mini_train[:sub_size, :, :, :]  # This is just a subset
mini_target = mini_target[:sub_size, :, :, :]

# Define some parameters of the network
img_shape = mini_train[0, :, :, :].shape

# ----------------------------------------------------------------------------------------------------------------------
# MODEL DEFINITION

print("Setting up the model")
base_n_filters = 32  # The "minimum" number of filters (filters in first encoder)
# Input
inputs = Input(shape=img_shape)
# Encoders
encoder0_pool, encoder0 = encoder_block(inputs, base_n_filters)  # 32 filters
encoder1_pool, encoder1 = encoder_block(encoder0_pool, base_n_filters*2)  # 64
encoder2_pool, encoder2 = encoder_block(encoder1_pool, base_n_filters*4)  # 128
# encoder3_pool, encoder3 = encoder_block(encoder2_pool, base_n_filters*8)  # 256 DEBUG this layers have been deleted
# to fit data size without recalculating better sizes
# encoder4_pool, encoder4 = encoder_block(encoder3_pool, base_n_filters*16)  # 512
print("Encoders created")
# Center
# center = conv_block(encoder4_pool, base_n_filters*32)  # 1024
center = conv_block(encoder2_pool, base_n_filters*8)  # 256
print("Center created")
# Decoders
# decoder4 = decoder_block(center, encoder4, base_n_filters*16)  # 512
# decoder3 = decoder_block(decoder4, encoder3, base_n_filters*8)  # 256
decoder2 = decoder_block(center, encoder2, base_n_filters*4)  # 128
# decoder2 = decoder_block(decoder3, encoder2, base_n_filters*4)  # 128
decoder1 = decoder_block(decoder2, encoder1, base_n_filters*2)  # 64
decoder0 = decoder_block(decoder1, encoder0, base_n_filters)  # 32
print("Decoders created")

# Output
outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
# Create model
model = Model(inputs=[inputs], outputs=[outputs])
# Compile model
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])

print("Ready to start training")
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss',
                                        save_best_only=True, verbose=1)
history = model.fit(x=mini_train,
                    y=mini_target,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[cp],
                    validation_split=0.2)
                    # TODO Consider shuffle

