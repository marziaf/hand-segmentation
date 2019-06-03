# Paths
from paths import *
# Matlab file reader
import scipy.io as sio
from scipy import ndimage

# Tensorflow
import tensorflow as tf
from keras import utils

# Plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random


# Uploads data from mat file to numpy array
# Reduce arrays sizes by reduce_images and reduction_factor
# Returns features, labels as numpy arrays
from scipy import ndimage


def get_data(feat_path, lab_path, feat_variable='features', lab_variable='labels',
                   reduce_images=False, reduction_factor=0.3):
    # Import data
    print("Importing data")
    features = sio.loadmat(feat_path).get(feat_variable)
    labels = sio.loadmat(lab_path).get(lab_variable)

    # Transpose in N x M x #Channels x #Set
    print("Transposing tensors")
    features = features.transpose(3, 0, 1, 2)  # 168x256x256x4
    labels = labels.transpose(3, 0, 1, 2).astype(int)
    num_img = int(features.shape[0])

    # If asked to, reduce the number of images
    if reduce_images:
        features = features[:int(num_img*reduction_factor), :, :, :]
        labels = labels[:int(num_img*reduction_factor), :, :, :]

    labels = utils.to_categorical(labels)

    return features, labels


def data_augmentation(feat, lab):  # TODO
    for i in range(0, int(feat.shape[0])):
        deg = random.randint(0, 359)
        feat[i, :, :, :] = ndimage.rotate(feat[i, :, :, :], deg, reshape=False)
        lab[i, :, :, :] = ndimage.rotate(lab[i, :, :, :], deg, reshape=False)


# Shows comparison between random couples of features and labels
def disp_some_data(feat, lab):
    print("sikfsedugbdffjb")
    # Get a "displayable" array
    lab = np.argmax(lab, axis=-1)

    fig_size = 10
    fig = plt.figure(figsize=(fig_size*2, fig_size))
    outer = gridspec.GridSpec(3, 3, wspace=0.2, hspace=0.2)
    tot_images = int(feat.shape[0])

    print("Ready")
    for i in range(9):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i],
                                                 wspace=0.1, hspace=0.1)
        inx = random.randint(0, tot_images-1)

        ax = fig.add_subplot(inner[0])
        ax.imshow(feat[inx, :, :, :3])
        ax.axis('off')
        ax = fig.add_subplot(inner[1])
        ax.axis('off')
        ax.imshow(lab[inx, :, :])
        fig.add_subplot(ax)

    print("ready to show")
    fig.show()

