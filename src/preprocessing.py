# # -----------------------------------------------------------------------------
# # Preprocessing File used in registration.py
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------


import numpy as np
import skimage

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, image):
        image = self.minmaxNormalization(image)
        image = self.clahe(image)
        return image
        
    @staticmethod
    def minmaxNormalization(image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    @staticmethod
    def clahe(image):
        # applies clahe to each axial slice of the image (assumes z,y,x)
        for i, axialSlice in enumerate(image):
            image[i] = skimage.exposure.equalize_adapthist(axialSlice)
        return image
