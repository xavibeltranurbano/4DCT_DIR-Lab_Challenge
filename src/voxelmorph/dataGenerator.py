# # -----------------------------------------------------------------------------
# # Data Generator For Voxelmorph
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------

from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
# Add the utils folder to sys.path
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessing import Preprocessing

class CTDataGenerator(Sequence):
    def __init__(self, inhalation_images, exhalation_images, batch_size=1):
        self.inhalation_images = inhalation_images
        self.exhalation_images = exhalation_images
        self.batch_size = batch_size
        self.preprocessing=Preprocessing()

    def __len__(self):
        # Since the batch size is 1, the length is equal to the number of images
        return len(self.inhalation_images)

    def __getitem__(self, idx):
        # Fetch a single batch of data
        inhalation_batch = self.inhalation_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        exhalation_batch = self.exhalation_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        inhalation_batch = self.preprocessing.preprocess(np.array(inhalation_batch))[...,np.newaxis]
        exhalation_batch = self.preprocessing.preprocess(np.array(exhalation_batch))[...,np.newaxis]
        zero_phi = np.zeros(inhalation_batch.shape)
        # Add any additional processing here if necessary
        return [exhalation_batch,inhalation_batch],[inhalation_batch,zero_phi]
    

