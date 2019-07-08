from paths import save_model_dir, test_features_path, test_labels_path, save_output_images_path
from os import path as op
from data_generator import get_data, disp_some_data
from network import ce_dice_loss
import keras
from keras import models, Model, utils
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model")
args = parser.parse_args()
model = args.model

model_path = op.join(save_model_dir, model)

# %% Load model
print("Loading model")
model = models.load_model(model_path, custom_objects={'ce_dice_loss': ce_dice_loss})
# %% Image acquisition
print("Getting test images")
test_features = np.load(model_path)
test_features = test_features[:30]  # TODO remove if memory saving is not needed
# test_labels = np.load(test_features_path)
# print("Getting categorical version of labels")
# test_labels = utils.to_categorical(test_labels)

# %% Test

# Prediction
print("Predicting labels")
prediction = model.predict(test_features)

# TODO aggiungere metriche oggettive per la valutazione

# del model  # TODO use if needed to free memory

disp_some_data(test_features, prediction, save_image=True, fig_size=20)