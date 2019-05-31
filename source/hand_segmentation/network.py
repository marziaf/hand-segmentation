# keras
from keras.layers import *
from keras.models import Model
from keras import losses
from keras import utils
from keras.utils import to_categorical

import tensorflow as tf


# NETWORK COMPONENTS

# Convolution block
def __conv_block(input_tensor, num_filters, kernel_size=3):
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
def __encoder_block(input_tensor, num_filters, pool_size=2, strides=2):
    # Encoder
    encoder = __conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling2D(pool_size=(pool_size, pool_size), strides=(strides, strides))(encoder)

    return encoder_pool, encoder


# Decoder block
def __decoder_block(input_tensor, concat_tensor, num_filters, kernel_size=3, transpose_kernel_size=2, strides=2):
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


# CUSTOM METRICS

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


def ce_dice_loss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


# BUILD MODEL

def get_unet_model(img_shape):
    print("Setting up the model")
    base_n_filters = 32  # The "minimum" number of filters (filters in first encoder)

    # Input
    inputs = Input(shape=img_shape)

    # Encoders
    encoder0_pool, encoder0 = __encoder_block(inputs, base_n_filters)  # 32 filters
    encoder1_pool, encoder1 = __encoder_block(encoder0_pool, base_n_filters*2)  # 64
    encoder2_pool, encoder2 = __encoder_block(encoder1_pool, base_n_filters*4)  # 128
    encoder3_pool, encoder3 = __encoder_block(encoder2_pool, base_n_filters*8)  # 256
    encoder4_pool, encoder4 = __encoder_block(encoder3_pool, base_n_filters*16)  # 512
    print("Encoders created")

    # Center
    center = __conv_block(encoder4_pool, base_n_filters*32)  # 1024
    print("Center created")

    # Decoders
    decoder4 = __decoder_block(center, encoder4, base_n_filters*16)  # 512
    decoder3 = __decoder_block(decoder4, encoder3, base_n_filters*8)  # 256
    decoder2 = __decoder_block(decoder3, encoder2, base_n_filters*4)  # 128
    decoder1 = __decoder_block(decoder2, encoder1, base_n_filters*2)  # 64
    decoder0 = __decoder_block(decoder1, encoder0, base_n_filters)  # 32
    print("Decoders created")

    # Output
    outputs = Conv2D(9, (1, 1), activation='softmax')(decoder0)

    # Create model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model
