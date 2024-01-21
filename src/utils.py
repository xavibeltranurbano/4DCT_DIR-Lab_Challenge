# -----------------------------------------------------------------------------
# Utils file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def allCallbacks(networkName):
        # Save weights of each epoch
        os.makedirs(f"/notebooks/results/{networkName}/1", exist_ok=True)
        pathWeights=f"/notebooks/results/{networkName}/1"
        checkpoint_path = pathWeights+"/epoch-{epoch:02d}.h5"
        model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False,
        verbose=0)

        # Reduce learning rate
        reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1,        # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=10,        # Number of epochs with no improvement after which learning rate will be reduced.
        min_lr=0.000001,    # Lower bound on the learning rate.
        verbose=1)

        # Early stopping callback
        early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=20,       # Number of epochs with no improvement after which training will be stopped.
        verbose=1)

        return model_checkpoint_callback,reduce_lr_callback,early_stopping_callback

    @staticmethod
    def save_training_plots(history, file_path):
        # Extract loss and Dice coefficient from the 'history' object
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_dice_coef = history.history['mean_dice_coef']
        val_dice_coef = history.history['val_mean_dice_coef']

        # Determine the actual number of epochs
        epochs = len(train_loss)
        
        # Create subplots for Loss and Dice coefficient
        plt.figure(figsize=(12, 5))

        # Plot training and validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation Dice coefficient
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_dice_coef, label='Training Dice Coefficient')
        plt.plot(range(1, epochs + 1), val_dice_coef, label='Validation Dice Coefficient')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Coefficient')
        plt.title('Training and Validation Dice Coefficient')
        plt.legend()

        # Save the plots to the specified file path
        plt.tight_layout()
        plt.savefig(file_path)
    
    
    
    