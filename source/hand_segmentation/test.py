from paths import save_model_path, test_features_path, test_labels_path, save_output_images_path
from data_generator import get_data, disp_some_data
from network import ce_dice_loss
import keras
from keras import models, Model

# %% Load model
print("Loading model")
model = models.load_model(save_model_path, custom_objects={'ce_dice_loss': ce_dice_loss})
# %% Image acquisition
print("Getting test images")
test_features, test_labels = get_data(test_features_path,  test_labels_path)
test_features = test_features[::4, :, :]  # TODO remove for real evaluations. Keep to preserve memory

# %% Test

# Prediction
print("Predicting labels")
prediction = model.predict(test_features)

# del model  # TODO use if needed to free memory

disp_some_data(test_features, prediction, save_image=True, fig_size=20)