# # -----------------------------------------------------------------------------
# # Main File to reproduce tests on the COPDgene dataset
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------

from utils import Utils
from registration import Registration
from evaluation import Evaluation
import os
import csv
import numpy as np

class COPDgene:
    evaluation = Evaluation()
    utils = Utils()

    def __init__(self, datasetDirectory, outputDirectory, parameterFolder):
        self.datasetDirectory = datasetDirectory
        self.outputDirectory = outputDirectory
        self.parameterFolder = parameterFolder
        self.evalResultsDirectory = "evaluation"
        self.utils.ensureFolderExists(self.evalResultsDirectory)

        # SPACINGS
        self.spacings = {
            "copd1" : [0.625, 0.625, 2.5],
            "copd2" : [0.645, 0.645, 2.5],
            "copd3" : [0.652, 0.652, 2.5],
            "copd4" : [0.590, 0.590, 2.5],
            "copd5" : [0.647, 0.647, 2.5],
            "copd6" : [0.633, 0.633, 2.5],
            "copd7" : [0.625, 0.625, 2.5],
            "copd8" : [0.586, 0.586, 2.5],
            "copd9" : [0.664, 0.664, 2.5],
            "copd10" : [0.742, 0.742, 2.5],
        }


        # initialize registration
        self.registration = Registration(
            self.parameterFolder, 
            outputDirectory = self.outputDirectory,
            usePreprocessing=False,
            storeTransformParameterMaps = True,
            storeImage = True,
            storePointFile = True,
            logToConsole=True
            )

    #################################
    ### REGISTRATION FUNCTIONS ######
    #################################

    def registerTrain(self, segmentation=False):
        imageNumbers =[1,2,3,4]
        for imageNumber in imageNumbers:
            paths = self.initRegistrationPathsDict(imageNumber, segmentation)
            self.registration.setOutputDirectory(paths["outputDirectory"])
            self.registration.register(paths["fixedImagePath"], paths["movingImagePath"], paths["pointFilePath"])


    def initRegistrationPathsDict(self, imageNumber, segmentation):
        # creates a path dict with inhale as moving and exhale as fixed
        pathsDict = {
            "pointFilePath": os.path.join(self.datasetDirectory, f"copd{imageNumber}/copd{imageNumber}_300_iBH_xyz_r1.txt"),
            "outputDirectory": os.path.join(self.outputDirectory, f"copd{imageNumber}")
        }

        if segmentation:
            pathsDict["fixedImagePath"] = os.path.join(self.datasetDirectory, f"copd{imageNumber}/segmentations/copd{imageNumber}_iBHCT_segmented.nii")
            pathsDict["movingImagePath"] = os.path.join(self.datasetDirectory, f"copd{imageNumber}/segmentations/copd{imageNumber}_eBHCT_segmented.nii")
        else:
            pathsDict["fixedImagePath"] = os.path.join(self.datasetDirectory, f"copd{imageNumber}/copd{imageNumber}_iBHCT.nii")
            pathsDict["movingImagePath"] = os.path.join(self.datasetDirectory, f"copd{imageNumber}/copd{imageNumber}_eBHCT.nii")

        return pathsDict

    #################################
    ### PREDICTION FUNCTIONS ########
    #################################

    def predictTrain(self):
        imageNumbers = [1,2,3,4]
        for imageNumber in imageNumbers:
            folderName = f"copd{imageNumber}"
            pointFilePath = os.path.join(self.outputDirectory, folderName, "outputpoints.txt")
            outputFilePath = os.path.join(self.outputDirectory, f"prediction_copd{imageNumber}.txt")
            self.evaluation.extractOutputPoints(pointFilePath, outputFilePath)


    #################################
    ### EVALUATION FUNCTIONS ########
    #################################
    
    def evaluateTrain(self, resultName):
        imageNumbers = [1, 2, 3, 4]
        treValues = []
        with open(os.path.join(self.evalResultsDirectory, f"tre_in_mm_{resultName}.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "TRE"])  

            for imageNumber in imageNumbers:
                pointSetPath1 = os.path.join(self.outputDirectory, f"prediction_copd{imageNumber}.txt")
                pointSetPath2 = os.path.join(self.datasetDirectory, f"copd{imageNumber}/copd{imageNumber}_300_eBH_xyz_r1.txt")


                pointSet1 = self.evaluation.readPointsFromFile(pointSetPath1)
                pointSet2 = self.evaluation.readPointsFromFile(pointSetPath2)

                normalizedPointSet1 = self.evaluation.normalizePoints(pointSet1, self.spacings[f"copd{imageNumber}"])
                normalizedPointSet2 = self.evaluation.normalizePoints(pointSet2, self.spacings[f"copd{imageNumber}"])

                tre = self.evaluation.targetRegistrationError(normalizedPointSet1, normalizedPointSet2)
                treValues.append(tre)
                writer.writerow([f"copd{imageNumber}", tre])

            writer.writerow([f"mean", np.mean(treValues)])
            writer.writerow([f"std", np.std(treValues)])





if __name__ == "__main__":
    # SETTINGS
    datasetDirectory = "data"
    outputDirectory = "results"
    parameterFolder = "customParameters"


    # FUNCTION CALLS
    copdgene = COPDgene(datasetDirectory, outputDirectory, parameterFolder)
    copdgene.registerTrain(segmentation=True)
    copdgene.predictTrain()
    copdgene.evaluateTrain("NAME_OF_THE_TEST")




