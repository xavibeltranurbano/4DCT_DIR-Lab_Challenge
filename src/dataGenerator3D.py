# -----------------------------------------------------------------------------
# Data Generator (3D) file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from MRI_preprocessing import Preprocessing

class BratsDataGenerator3D(tf.keras.utils.Sequence):
    def __init__(self, datasetFolder, batchSize=1, targetDimensions=(256, 128, 256), numChannels=1, numClasses=1, shuffle=True, isTraining=True):
        self.targetDimensions = targetDimensions
        self.batchSize = batchSize
        self.datasetFolder = datasetFolder
        self.numChannels = numChannels
        self.shuffle = shuffle
        self.isTraining = isTraining
        self.numClasses = numClasses
        self.preprocessing = Preprocessing(data_aug=False, norm_intensity=True)
        self.fileList = self._getFileList()
        self.onEpochEnd()

    def _getFileList(self):
        subsetFolder = 'Training_Set' if self.isTraining else 'Validation_Set'
        filePaths = []
        path = os.path.join(self.datasetFolder, subsetFolder)
        for folderName in os.listdir(path):
            imagePath = os.path.join(path, folderName, folderName + '.nii.gz')
            maskPath = os.path.join(path, folderName, folderName + '_seg.nii.gz')
            if os.path.isfile(imagePath) and os.path.isfile(maskPath):
                filePaths.append((imagePath, maskPath))
        return filePaths

    def __len__(self):
        return int(np.floor(len(self.fileList) / self.batchSize))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batchSize:(index + 1) * self.batchSize]
        batchFiles = [self.fileList[k] for k in indexes]
        images, masks = self._generateImagesAndMasks(batchFiles)
        return images, masks

    def onEpochEnd(self):
        self.indexes = np.arange(len(self.fileList))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generateImagesAndMasks(self, batchFiles):
        images = np.empty((self.batchSize, *self.targetDimensions, self.numChannels))
        masks = np.empty((self.batchSize, *self.targetDimensions, self.numClasses))

        for i, (imagePath, maskPath) in enumerate(batchFiles):
            image = nib.load(imagePath).get_fdata()
            mask = nib.load(maskPath).get_fdata()
            # preprocessing
            image = self.preprocessing.normalizeIntensity(image)
            converted_mask = self.convertToOneHot(mask.squeeze(-1))

            images[i, ] = image
            masks[i, ] = converted_mask        
            
        return images, masks

    def convertToOneHot(self, mask):
        oneHotEncodedMask = np.zeros((*mask.shape, self.numClasses))

        # For each class, set the corresponding channel to 1 where the mask equals the class value
        for i in range(self.numClasses):
            oneHotEncodedMask[..., i] = (mask == i).astype(np.int)

        return oneHotEncodedMask
