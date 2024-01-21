# -----------------------------------------------------------------------------
# Main (2D) file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

from utils import Utils
from networks.unet import Unet
from networks.segnet import SegNet
from networks.segresnet import SegResNet
import os
from metrics import *
from keras.optimizers import Adam
from configuration import Configuration
import numpy as np
from keras import backend as K




def run_program(config,networkName, params):
    # Clear any existing TensorFlow session
    K.clear_session()
    utils = Utils()
    # Generate the IDs for train, val and test
    trainGenerator, valGenerator=config.createAllDataGenerators()
    network = Unet(img_rows=params['targetSize'][0], img_cols=params['targetSize'][0], channels=1, classes=params['nClasses'])
    model = network.get_model()
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0004),loss=categoricalCrossentropy,metrics=[mean_dice_coef,csf,gm,wm,bckgrd])
    #model.load_weights('/notebooks/results/Unet/1/epoch-40.h5',by_name=True, skip_mismatch=True)
    with tf.device('/GPU:0'):
        # Train the model
        model_checkpoint_callback,reduce_lr_callback,early_stopping_callback = utils.allCallbacks(networkName)
        epochs = 300
        history = model.fit(trainGenerator, validation_data=valGenerator, epochs=epochs, verbose=1,
                           callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stopping_callback])

        model.save(f"/notebooks/results/{networkName}/1/Best_Model.h5")
        # Plot the results and save the image
        utils.save_training_plots(history, f"/notebooks/results/{networkName}/training_plots.png")
        # Predict test set
        loss,acc=model.evaluate(valGenerator,verbose=1)
        print(f"\nTest: Dice= {acc}, Loss= {loss}")
    
if __name__ == "__main__":
    imgPath = '/notebooks/data'
    # Nertwork Unet
    networkName="SegResNet"
    
    # Parameters of the training
    params={
        'pathData':imgPath,
        'targetSize':(256,256,1),
        'batchSize':1,
        'miniBatchSize' :88,
        'nClasses':4
    }
    
    # Create folder for this experiment
    os.makedirs(f"/notebooks/results/{networkName}_##PROVAv3", exist_ok=True)
    
    # Configuration of the experiment
    config=Configuration(**params)
    
    #Run experiment
    run_program(config,networkName,params)
