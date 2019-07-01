# Custom
from data_generator import *
from network import *
from paths import *

# Params
import argparse

# Keras models
from keras import models

# %% Input parsing
# TODO: fix parser adding other inputs (e.g. learning rate, optimizer,...)
parser = argparse.ArgumentParser()
# TODO optimize parameters
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=15)
parser.add_argument("--patience", type=int, default=10)

args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
patience = args.patience


# %% Obtain data
features, labels = get_data(features_path, labels_path, reduce_images=True, reduction_factor=0.06)  # TODO don't reduce if possible
im_size = features.shape[1:4]

# %% Model

# Get model
model = get_unet_model(im_size)

# Compile
# As metrics we would like the pixel accuracy rather than the loss.
# Adam is ok, you might want to try other optimizers (e.g. SGD, Adagrad/Adadelta...) and different learning rates.
# To specify the lr, need to create an optimizer object TODO
model.compile(optimizer='adam', loss=ce_dice_loss, metrics=['accuracy'])

# Train
print("Ready to start training")
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, save_best_only=True, verbose=1)

# Callbacks

# TODO: visualize learning via tensorboard (tensorboard --logdir 'tb/xxx' --port 6001)
callbacktb = tf.keras.callbacks.TensorBoard(log_dir="tb",
                                            histogram_freq=0,
                                            write_graph=True,
                                            write_images=False,
                                            update_freq='batch')

early_stop = tf.keras.callbacks.EarlyStopping(patience=patience,
                                              min_delta=0,
                                              restore_best_weights=True)

callbacks = [callbacktb, early_stop, cp]

# Fit

model.fit(x=features,
          y=labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks,
          validation_split=0.2,
          shuffle=True)

# %% TODO: Evaluate performance  (sample code to modify)
# y_hat = m.predict(x_test)

# m.save(path.join(save_path, "model.h5"))
# np.save(path.join(savepath, "history"), history)
# np.save(path.join(save_path, "y_hat"), y_hat)
# utils.write_results(y_true=y_test,
#                    y_pred=y_hat,
#                    filepath=path.join(save_path, "results.txt"))


# %%
# TODO: sanity check on losses (I had negative values)
# TODO: compare results using CE, CE+dice, dice
# TODO: generate more data (maybe then need to use a data generator which does not load all the images in RAM at once)
#  --> maybe we ask professor about source code

# TODO: compute per class pixel accuracy, per class IoU, mean IoU, mean PA, mean class accuracy (using provided file)

