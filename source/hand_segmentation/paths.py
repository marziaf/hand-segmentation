from os import path as op

# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# train features and labels paths
sets_root_path = op.join(data_root_path, 'sets')
train_features_path = op.join(sets_root_path, 'train_feat.npy')
train_labels_path = op.join(sets_root_path, 'train_lab.npy')
# validation
validation_features_path = op.join(sets_root_path, 'val_feat.npy')
validation_labels_path = op.join(sets_root_path, 'val_lab.npy')
# test sets
test_features_path = op.join(sets_root_path, 'test_feat')
test_labels_path = op.join(sets_root_path, 'test_lab')
# output model paths
model_id = 'new_train_try'
save_model_dir = op.join(project_root_path, 'models')
save_model_path = op.join(save_model_dir, model_id+'.hdf5')  # TODO check existing files and name progressively
# test image outputs
save_output_images_path = op.join(project_root_path, 'outputs/', model_id)
