from os import path as op

# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# features and labels paths
sets_root_path = op.join(data_root_path, 'sets')
features_path = op.join(sets_root_path, 'features.mat')
labels_path = op.join(sets_root_path, 'labels.mat')
# output model paths
save_model_dir = op.join(project_root_path, 'models')
save_model_path = op.join(save_model_dir, 'third_try.hdf5')

