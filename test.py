# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model

# Helper libraries

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Images

import imageio
from matplotlib import pyplot as plt

# Paths
import os
from os import path as op

# PATHS
project_root_path = op.relpath('..')
data_root_path = op.join(project_root_path, 'data')
# mini
mini_root_path = op.join(data_root_path, 'mini')
mini_all_features_path = op.join(mini_root_path, 'all_features')
mini_target_path = op.join(mini_root_path, 'target')

# Get data/home/marzia
print("Getting data")
mini_train = sio.loadmat(op.join(mini_all_features_path, 'features_train.mat')).get('features_train')
mini_target = sio.loadmat(op.join(mini_target_path, 'target_train.mat')).get('target_train')


# Transpose to get NxMx4
print("Transposing tensors")
mini_train = mini_train.transpose(3, 1, 2, 0)  # 126x200x200x4
mini_target = mini_target.transpose(3, 1, 2, 0)

# Define some parameters of the network
input_shape = mini_train[:,:,:,0].shape
batch_size = 2  # random choice
epochs = 3  # random choice


# Network test


