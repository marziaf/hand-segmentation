from paths import save_model_dir, test_features_path, test_labels_path, save_output_images_path
from os import path as op
from data_generator import get_data, disp_some_data
from network import ce_dice_loss
import keras
from keras import models, Model, utils
from sklearn import metrics
import numpy as np
import argparse
from myMetrics import compute_and_print_IoU_per_class
import random

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
test_features = np.load(test_features_path)

num_images = 10
fr = random.randint(0,test_features.shape[0]-num_images)

test_features = test_features[fr:fr+num_images]  # TODO remove if memory saving is not needed
test_labels = np.load(test_labels_path)[fr:fr+num_images]

print("Getting categorical version of labels")
test_labels = utils.to_categorical(test_labels)


# Prediction
print("------------------------------------------------Predicting labels------------------------------------------")
prediction = model.predict(test_features)

# TODO aggiungere metriche oggettive per la valutazione

# del model  # TODO use if needed to free memory

#disp_some_data(test_features, prediction, fig_size=20)
#input()

print("----------Getting IoU--------------")
num_cl = test_labels.shape[-1]
for i in range(num_images):  # run over different classes
    im1 = test_labels[i]
    im2 = prediction[i]

    im1 = im1.reshape(256*256, 8)
    im2 = im2.reshape(256*256, 8)

    confusion_matrix = metrics.confusion_matrix(im1.argmax(axis=-1), im2.argmax(axis=-1))
    compute_and_print_IoU_per_class(confusion_matrix, 8, class_mask=None)