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
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=15)
parser.add_argument("--patience", type=int, default=10)

args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
patience = args.patience


# %% Obtain data
print("--------Getting data--------")
train_f, train_l, val_f, val_l = get_data(train_path_feat=train_features_path,
                                          train_path_lab=train_labels_path,
                                          val_path_feat=validation_features_path,
                                          val_path_lab=validation_labels_path,
                                          reduce_images=True,
                                          reduction_factor=0.5)  # TODO don't reduce if possible
# Get the size of the images
im_size = train_f.shape[1:4]

# %% Model

# Get model
print("-----Getting network model-----")
model = get_unet_model(im_size)

# Compile
print("-----Compiling-----")
# As metrics we would like the pixel accuracy rather than the loss.
# Adam is ok, you might want to try other optimizers (e.g. SGD, Adagrad/Adadelta...) and different learning rates.
# To specify the lr, need to create an optimizer object TODO
model.compile(optimizer='adam', loss=ce_dice_loss, metrics=['accuracy'])

# Train
print("-------Ready to start training-------")
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

print("------Start fitting the model------")

model.fit(x=train_f,
          y=train_l,
          validation_data=(val_f, val_l),
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks,
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
# TODO: compare results using CE, CE+dice, dice

# TODO: compute per class pixel accuracy, per class IoU, mean IoU, mean PA, mean class accuracy (using provided file)

