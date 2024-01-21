# # -----------------------------------------------------------------------------
# # Training Of Voxelmorph
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------


import os
import voxelmorph as vxm
import tensorflow as tf
from dataGenerator import CTDataGenerator
from utils_vxm import Utils


def run_program(imgPath):
    utils=Utils(imgPath)
    # We read the dataset
    print("Reading images...")
    inhalation_images,exhalation_images = utils.load_data()
    train_generator = CTDataGenerator(inhalation_images, exhalation_images, batch_size=1)
    
    # Set parameters and train model
    print("Starting the training...")
    vol_shape = (256, 256, 128)
    nb_features = [[16, 32, 32, 32],[32, 32, 32, 32, 32, 16, 16]]
    lossCombination = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    loss_weights = [1, 0.01]
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
    vxm_model.compile(tf.keras.optimizers.Adam(lr=1e-3), loss=lossCombination,loss_weights=loss_weights)
    vxm_model.load_weights("/notebooks/voxelmorph/Best_Model.h5")
    history=vxm_model.fit(train_generator, epochs=100)
    # Save Model
    vxm_model.save("/notebooks/voxelmorph/Best_Model.h5")
    # Plot results
    utils.save_training(history, filename='/notebooks/voxelmorph/training_plot.png')

    # Predict all the images and compute final metrics
    utils.compute_Metrics(vxm_model)
    
if __name__ == "__main__":
    imgPath = 'data'
    run_program(imgPath)
