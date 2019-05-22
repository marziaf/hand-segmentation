# TensorFlow and tf.keras

from keras.layers import *
from keras.models import Model

# Helper libraries

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Images

import imageio
from matplotlib import pyplot as plt

# Paths
from os import path as op


# Network components
def conv_block(input_tensor, num_filters):
    encoder = Conv2D(num_filters, (2, 2), padding='same')(input_tensor)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    encoder = Conv2D(num_filters, (2, 2), padding='same')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling2D((3, 3), strides=(2, 2))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = concatenate([concat_tensor, decoder], axis=-1)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2D(num_filters, (2, 2), padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    decoder = Conv2D(num_filters, (2, 2), padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Activation('relu')(decoder)
    return decoder


# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# mini
mini_root_path = op.join(data_root_path, 'mini')
mini_all_features_path = op.join(mini_root_path, 'all_features')
mini_target_path = op.join(mini_root_path, 'target')

# Get data
print("Getting data")
mini_train = sio.loadmat(op.join(mini_all_features_path, 'features_train.mat')).get('features_train')
mini_target = sio.loadmat(op.join(mini_target_path, 'target_train.mat')).get('target_train')


# Transpose to get NxMx4
print("Transposing tensors")
mini_train = mini_train.transpose(3, 1, 2, 0)  # 126x200x200x4
mini_target = mini_target.transpose(3, 1, 2, 0)

# Define some parameters of the network
img_shape = mini_train[:, :, :, 0].shape
batch_size = 2  # random choice
epochs = 3  # random choice

# Network
inputs = Input(shape=img_shape)

encoder0_pool, encoder0 = encoder_block(inputs, 16)
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
print("Encoders created")
center = conv_block(encoder1_pool, 64)
print("Center created")
decoder1 = decoder_block(center, encoder1, 32)
decoder0 = decoder_block(decoder1, encoder0, 16)
print("Decoders created")

# outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

# model = Model(inputs=mini_train, outputs=mini_target)
