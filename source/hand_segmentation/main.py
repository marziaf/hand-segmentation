# Custom
from data_generator import *
from network import *

# Paths
from os import path as op

# Keras models
from keras import models


# PATHS
project_root_path = op.relpath('../..')
data_root_path = op.join(project_root_path, 'data')
# training/validation features and labels paths
sets_root_path = op.join(data_root_path, 'sets')
features_path = op.join(sets_root_path, 'features.mat')
labels_path = op.join(sets_root_path, 'labels.mat')
# test paths
test_features_path = op.join(sets_root_path, 'test_features.mat')
test_labels_path = op.join(sets_root_path, 'test_labels.mat')

# output model paths
save_model_dir = op.join(project_root_path, 'models')
save_model_path = op.join(save_model_dir, 'third_try.hdf5')

# Obtain data
features, labels = get_data(features_path, labels_path, reduce_images=True, reduction_factor=0.1)
im_size = features.shape[1:4]

# Get model
model = get_unet_model(im_size)

# Compile
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])

# Train
print("Ready to start training")
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss',
                                        save_best_only=True, verbose=1)

batch_size = 3
epochs = 5
history = model.fit(x=features,
                    y=labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[cp],
                    validation_split=0.2)
