# # -----------------------------------------------------------------------------
# # Registration File
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------

import itk # itk-elastix
import numpy as np
import os
import logging

from utils import Utils
from preprocessing import Preprocessing

class Registration:

    util = Utils()

    def __init__(self, parameterFolder, outputDirectory="outputDirectory", usePreprocessing=False, storeTransformParameterMaps=True, storeImage=True, storePointFile=False, logToConsole=False):
        # SETTINGS
        self.parameterFolder = parameterFolder
        self.outputDirectory = outputDirectory
        self.storeTransformParameterMaps = storeTransformParameterMaps
        self.storeImage = storeImage
        self.storePointFile = storePointFile

        # FUNCTION CALLS
        self.initLogging(logToConsole)
        self.initPreprocessing(usePreprocessing)
        self.util.ensureFolderExists(self.outputDirectory)


    #################################
    ### REGISTRATION ################
    #################################

    def initParamaterObject(self, parameterFolder):
        # initializes a parameter object with the registration in the parameter Folder. Automatically sorted after rigid -> affine -> bspline
        logging.info(f"Initialiazing Parameters from {parameterFolder}.")
        parameterObject = itk.ParameterObject.New()
        parameterMaps = self.util.getAllFiles(parameterFolder)
        sortedMaps = sorted(parameterMaps, key=self.util.getRegistrationSortKey)

        registrationTypeList = []
        for i, parameterPath in enumerate(sortedMaps):
            parameterObject.AddParameterFile(parameterPath)
            logging.info(f"Successfully loaded {parameterPath}. Application order: {i+1}.")
            registrationTypeList.append(os.path.basename(parameterPath).split(".")[0])
        return parameterObject, registrationTypeList

    def register(self, fixedImagePath, movingImagePath, pointFilePath=None):
        # registers an image
        fixedImage = self.util.loadImageFrom(fixedImagePath)
        movingImage = self.util.loadImageFrom(movingImagePath)
        parameterObject, self.registrationTypeList = self.initParamaterObject(self.parameterFolder)

        if self.usePreprocessing:
            logging.info(f"Applying Preprocessing.")
            fixedImage = self.applyPreprocessing(fixedImage)
            movingImage = self.applyPreprocessing(movingImage)

        logging.info(f"registering {movingImagePath} to {fixedImagePath}.")
        resultImage, resultTransformParameters = itk.elastix_registration_method(fixedImage, movingImage, parameter_object=parameterObject, log_to_console=True)
        logging.info(f"registered {movingImagePath} to {fixedImagePath}.")


        if self.storeTransformParameterMaps:
            self.safeTransformParameterObject(resultTransformParameters, movingImagePath)

        if self.storeImage:
            self.safeImage(resultImage, movingImagePath)

        if self.storePointFile:
            self.safeTransformedPointFile(pointFilePath, movingImage, resultTransformParameters)

    #################################
    ### PREPROCESSING ###############
    #################################

    def initPreprocessing(self, usePreprocessing):
        # initializes the preprocessing class
        self.usePreprocessing = usePreprocessing
        if usePreprocessing:
            self.preprocessing = Preprocessing()

    def applyPreprocessing(self, image):
        # takes and ITK image, applies preprocessing and returns an ITK image
        data = itk.GetArrayFromImage(image)
        data = self.preprocessing.preprocess(data)
        processedImage = itk.GetImageFromArray(data)
        return processedImage


    #################################
    ### STORAGE FUNCTIONS ###########
    #################################

    def safeTransformParameterObject(self, resultTransformParameters, movingImagePath):
        # saves the computed registration parameter files
        nParameterMaps = resultTransformParameters.GetNumberOfParameterMaps()
        imageName, _ = self.util.splitNameFromExtension(movingImagePath)

        for index in range(nParameterMaps):       
            fileName = imageName + "_" + self.registrationTypeList[index] + ".txt"
            folderPath = os.path.join(self.outputDirectory, "transformParameterMaps")
            self.util.ensureFolderExists(folderPath)
            finalPath = os.path.join(folderPath, fileName)
            parameterMap = resultTransformParameters.GetParameterMap(index)

            if index == nParameterMaps - 1:
                parameterMap['FinalBSplineInterpolationOrder'] =  "0"

            resultTransformParameters.WriteParameterFile(parameterMap, finalPath)
            logging.info(f"Saved Parametermap as {fileName} in {folderPath}.")


    def safeImage(self, image, movingImagePath):
        # safes image in output directory
        name, extension = self.util.splitNameFromExtension(movingImagePath)
        imageName = name + "_registered" + extension
        imagePath = os.path.join(self.outputDirectory, imageName)
        itk.imwrite(image, imagePath)
        logging.info(f"Saved registed image as {imageName} in {self.outputDirectory}.")


    def safeTransformedPointFile(self, pointFilePath, movingImage, transformParameterObject):
        # Apply the transformation to the point data 
        self.checkPointFile(pointFilePath)
        transformedPointFile = itk.transformix_pointset(
            movingImage, transformParameterObject,
            fixed_point_set_file_name=pointFilePath,
            output_directory = self.outputDirectory
            )
        logging.info(f"Saved point file as outputpoints.txt in {self.outputDirectory}.")


    ################################
    ### HELPER FUNCTIONS ###########
    ################################

    @staticmethod
    def checkPointFile(pointFilePath):
        # checks if the point file has a valid format
        if not pointFilePath:
            raise ValueError("The path to the point file (the file to be transformed) has not been set. In the case of registration use registration.register(fixedImagePath, movingImagePath, pointFilePath)")         
        try:
            with open(pointFilePath, 'r') as file:
                lines = file.readlines()
                if not "index" in lines[0].lower() or not lines[1].strip().isdigit():
                    raise ValueError("Incorrect format of the point file. The first line must contain the word point. The second the number of points. d"
                                        "A correct file could look like this:\n"
                                        "---pointFile.txt\n"
                                        "index\n"
                                        "3\n"
                                        "102.8 -33.4 57.0\n"
                                        "178.1 -10.9 14.5\n"
                                        "180.4 -18.1 78.9\n"
                                        "---\n"
                                        "For more information refer to https://simpleelastix.readthedocs.io/PointBasedRegistration.html"
                                        )

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Unable to find the file: {pointFilePath}") from e

        except Exception as e:
            raise Exception(f"An error occurred while reading the file: {pointFilePath}: {e}") from e

    def setOutputDirectory(self, newOutputDirectory):
        # sets outputDirectory
        self.outputDirectory = newOutputDirectory

    @staticmethod
    def initLogging(isEnabled):
        # initializes logs
        level = logging.INFO if isEnabled else logging.CRITICAL + 1
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')





if __name__ == "__main__":
    # SETTINGS
    parameterFolder = "PATH/TO/PARAMETERFOLDER"
    outputDirectory = "PATH/TO/outputDirectory"
    pointFilePath = "PATH/TO/POINTFILE.txt"
    fixedImagePath = "PATH/TO/FIXEDIMAGE.nii"
    movingImagePath = "PATH/TO/MOVINGIMAGE.nii"
    
    # REGISTRATION WITH SETTINGS
    registration = Registration(
            parameterFolder, 
            outputDirectory = outputDirectory,
            usePreprocessing=False,
            storeTransformParameterMaps = True,
            storeImage = True,
            storePointFile = True,
            logToConsole=True
            )
    registration.register(fixedImagePath, movingImagePath, pointFilePath)


    
        