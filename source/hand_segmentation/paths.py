from os import path as op

# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# features and labels paths
sets_root_path = op.join(data_root_path, 'sets')
features_path = op.join(sets_root_path, 'features.mat')
labels_path = op.join(sets_root_path, 'labels.mat')
# test sets
test_features_path = op.join(sets_root_path, 'test_features.mat')
test_labels_path = op.join(sets_root_path, 'test_labels.mat')
# output model paths
model_id = 'fourth_try'
save_model_dir = op.join(project_root_path, 'models')
save_model_path = op.join(save_model_dir, model_id+'.hdf5')  # TODO check existing files and name progressively
# test image outputs
save_output_images_path = op.join(project_root_path, 'outputs/', model_id)
