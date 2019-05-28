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
def __get_np_array(feat_path, lab_path, feat_variable='features', lab_variable='labels',
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


# Returns a tensorflow.data
# TODO is this passage of functions a waste of memory?
def get_data(feat_path, lab_path, reduce_set=False, reduction_factor=0.3):
    features, labels = __get_np_array(feat_path, lab_path, reduce_images=reduce_set,
                                      reduction_factor=reduction_factor)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # TODO shuffle...
    return dataset


# Displays random images couples of data1 and data2
def disp_data_comparison(dataset, num_rows=3, num_cols=3):

    data_iter = data.make_one_shot_iterator()
    next_element = data_iter.get_next()

    fig = plt.figure(figsize=(20, 10))
    outer = gridspec.GridSpec(num_rows, num_cols, wspace=0.2, hspace=0.2)
    tot_images = int(dataset.shape[0])

    for i in range(num_rows*num_cols):
        imgs, label = tf.keras.backend.get_session().run(next_element)
        img = imgs[0]
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        inx = random.randint(0, tot_images-1)

        ax = fig.add_subplot(inner[0])
        ax.imshow(dataset1[inx, :, :, :])
        ax.axis('off')
        ax = fig.add_subplot(inner[1])
        ax.axis('off')
        ax.imshow(dataset2[inx, :, :, :])
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
# f, l = __get_np_array(features_path, labels_path, reduce_images=True)
data = get_data(features_path, labels_path)
#disp_data_comparison(data)
