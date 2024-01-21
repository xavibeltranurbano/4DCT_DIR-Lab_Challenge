# -----------------------------------------------------------------------------
# Main (3D) file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plts

from networks.unet3d import Unet3D
from networks.segresnet import SegResNet
from dataGenerator3D import BratsDataGenerator3D
from utils import Utils
from metrics import *


def run_program(networkName, params):
    # Clear any existing TensorFlow session
    K.clear_session()
    utils = Utils()
    
    # Generate the IDs for train, val and test
    # Example usage
    trainGenerator = BratsDataGenerator3D(params["datasetFolder"], numClasses=params["nClasses"], isTraining=True)
    valGenerator = BratsDataGenerator3D(params["datasetFolder"], numClasses=params["nClasses"],isTraining=False)

    # Define model and compile
    model = SegResNet(input_shape=(256,128,256,1), spatial_dims=3, init_filters=16, in_channels=1, out_channels=4)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0004),loss=categoricalCrossentropy,metrics=[mean_dice_coef,csf,gm,wm,bckgrd])
    #model.load_weights('/notebooks/results/Unet_Coronal/1/Best_Model.h5',by_name=True, skip_mismatch=True)
    networkName=networkName+"_#####3D"
    
    
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

        # Send email with the results
        utils.sendEmailResults(acc,loss)
    
if __name__ == "__main__":
    datasetFolder = '/notebooks/data'
    # Nertwork Unet
    networkName="SegResNet"
    
    # Parameters of the training
    params={
        'datasetFolder':datasetFolder,
        'targetSize':{
            "img_depth" : 256,
            "img_rows" : 128,
            "img_cols" : 256,
            "channels" : 1,
            },
        'batchSize':1,
        'nClasses':4
    }
    
    # Create folder for this experiment
    os.makedirs(f"/notebooks/results/{networkName}_#####3D", exist_ok=True)
    
    #Run experiment
    run_program(networkName,params)
