# Custom
import data_generator
import network

# Paths
from os import path as op


# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# mini
sets_root_path = op.join(data_root_path, 'sets')
features_path = op.join(sets_root_path, 'features.mat')
labels_path = op.join(sets_root_path, 'labels.mat')

# Obtain data
features, labels = data_generator.get_data(features_path, labels_path, reduce_images=True)
im_size = features.shape[1:4]
model = network.get_unet_model(im_size)