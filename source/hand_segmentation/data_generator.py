# Paths
from paths import *

# keras
from keras import utils

# Plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# Uploads data as numpy array
# Reduce arrays sizes by reduce_images and reduction_factor
# Returns features, labels as numpy arrays


def get_data(train_path_feat, val_path_feat, train_path_lab, val_path_lab,
                   reduce_images=False, reduction_factor=0.3):
    # Import data
    print("-----------Importing data-----------")
    train_features = np.load(train_path_feat)
    print("Imported train set features")
    validation_features = np.load(val_path_feat)
    print("Imported validation set features")
    train_labels = np.load(train_path_lab)
    print("Imported train labels")
    validation_labels = np.load(val_path_lab)
    print("Imported validation labels")

    num_img_train = int(train_features.shape[0])
    num_img_val = int(validation_features.shape[0])

    print("--------Data manipulation------")
    # If asked to, reduce the number of images
    if reduce_images:
        print("Reducing train_features")
        train_features = train_features[:int(num_img_train*reduction_factor), :, :, :]
        print(train_features.shape)  # debug
        print("Reducing train_labels")
        train_labels = train_labels[:int(num_img_train*reduction_factor), :, :]
        print(train_labels.shape)  # debug
        print("Reducing validation_features")
        validation_features = validation_features[:int(num_img_val * reduction_factor), :, :, :]
        print(validation_features.shape)  # debug
        print("Reducing validation_labels")
        validation_labels = validation_labels[:int(num_img_val * reduction_factor), :, :]
        print(validation_labels.shape)  # debug


    print("Cathegorizing labels")
    train_labels = utils.to_categorical(train_labels)
    validation_labels = utils.to_categorical(validation_labels)

    print("-----------Data generated-------")
    return train_features, train_labels, validation_features, validation_labels


# Shows comparison between random couples of features and labels
def disp_some_data(feat, lab, save_image=False, save_path=save_output_images_path, fig_size=30):
    # Get a "displayable" array
    lab = np.argmax(lab, axis=-1)

    fig = plt.figure(figsize=(fig_size*2, fig_size))
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
        ax.imshow(lab[inx, :, :])
        fig.add_subplot(ax)

    fig.show()

    if save_image:
        fig.savefig(save_path)
