# Paths
from os import path as op
# Matlab file reader
import scipy.io as sio

# Tensorflow
import tensorflow as tf

# Plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random


# Uploads data from mat file to numpy array
# Reduce arrays sizes by reduce_images and reduction_factor
# Returns features, labels as numpy arrays
def get_data(feat_path, lab_path, feat_variable='features', lab_variable='labels',
                   reduce_images=False, reduction_factor=0.3):
    # Import data
    print("Importing data")
    features = sio.loadmat(feat_path).get(feat_variable)
    labels = sio.loadmat(lab_path).get(lab_variable)

    # Transpose in N x M x #Channels x #Set
    print("Transposing tensors")
    features = features.transpose(3, 0, 1, 2)  # 168x256x256x4
    labels = labels.transpose(3, 0, 1, 2)
    num_img = int(features.shape[0])

    # If asked to, reduce the number of images
    if reduce_images:
        features = features[:int(num_img*reduction_factor), :, :, :]
        labels = labels[:int(num_img*reduction_factor), :, :, :]

    return features, labels


def disp_some_data(feat, lab):

    fig = plt.figure(figsize=(20, 10))
    outer = gridspec.GridSpec(3, 3, wspace=0.2, hspace=0.2)
    tot_images = int(feat.shape[0])

    for i in range(9):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i],
                                                 wspace=0.1, hspace=0.1)
        inx = random.randint(0, tot_images-1)

        ax = fig.add_subplot(inner[0])
        ax.imshow(feat[inx, :, :, :3])
        ax.axis('off')
        ax = fig.add_subplot(inner[1])
        ax.axis('off')
        ax.imshow(lab[inx, :, :, 0])
        fig.add_subplot(ax)

    fig.show()


# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# mini
sets_root_path = op.join(data_root_path, 'sets')
features_path = op.join(sets_root_path, 'features.mat')
labels_path = op.join(sets_root_path, 'labels.mat')

# TEST
f, l = get_data(features_path, labels_path, reduce_images=True)
