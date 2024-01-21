# -----------------------------------------------------------------------------
# Configuration file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import os
from dataGenerator import DataGenerator

class Configuration():
    def __init__(self,pathData,targetSize,batchSize,miniBatchSize,nClasses):
        self.pathData=pathData
        self.targetSize=targetSize
        self.batchSize=batchSize
        self.miniBatchSize =miniBatchSize
        self.nClasses=nClasses
        self.kfold=None
        self.allIDS = [filename for filename in os.listdir(self.pathData + "/Training_Set") if filename != ".DS_Store" and ".ipynb_checkpoints" not in filename]

    def createDataGenerator(self, listIDS,setType, dataAugmentation,shuffle):
        # Create the Data Generator
        pathSet=os.path.join(self.pathData,setType )
        data_generator = DataGenerator(
            image_directory=pathSet,
            list_IDs=listIDS,
            batch_size=self.batchSize,
            target_size=self.targetSize,
            minibatch_size=self.miniBatchSize,
            data_augmentation=dataAugmentation,
            n_classes=self.nClasses,
            shuffle=shuffle)
        return data_generator

    def createAllDataGenerators(self):
        trainingIDS=['IBSR_01','IBSR_03','IBSR_04','IBSR_05','IBSR_06','IBSR_07','IBSR_08','IBSR_09','IBSR_16','IBSR_18']
        validationIDS=['IBSR_11','IBSR_12','IBSR_13','IBSR_14','IBSR_17']
        train_generator = self.createDataGenerator(trainingIDS, setType="Training_Set", dataAugmentation=True, shuffle=True)
        validation_generator = self.createDataGenerator(validationIDS, setType="Training_Set", dataAugmentation=False, shuffle=False)
        return train_generator, validation_generator

    