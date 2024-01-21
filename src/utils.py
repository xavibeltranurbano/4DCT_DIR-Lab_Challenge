# # -----------------------------------------------------------------------------
# # Utility File Containing Helper Functions
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------

import itk
import os
import pathlib
import nibabel as nib

class Utils:
    def __init__self(self):
        pass

    @staticmethod
    def loadImageFrom(imagePath):
        # Load images with itk floats (itk.F). Necessary for elastix
        return itk.imread(imagePath, itk.SS)

    @staticmethod
    def loadTransformParameterObject(filePaths):
        # initializes 
        parameterObject = itk.ParameterObject.New()
        for parameterPath in filePaths:
            parameterObject.AddParameterFile(parameterPath)
        return parameterObject

    @staticmethod
    def getAllFiles(folderPath):
        allImagePaths = os.listdir(folderPath)
        relativePaths = []
        for imagePath in allImagePaths:
            newPath = os.path.join(folderPath, imagePath)
            relativePaths.append(newPath)
        return relativePaths

    def splitFixedFromMoving(self, relativePaths, fixedName):
        fixedImageIndex = self.getImageIndex(relativePaths, fixedName)
        fixedImagePath = relativePaths.pop(fixedImageIndex)
        return fixedImagePath, relativePaths

    @staticmethod
    def getImageIndex(relativePaths, imageName):
        index = 0
        for relativePath in relativePaths:
            if imageName in relativePath:
                return index
            else:
                index += 1
        raise ValueError(f"{imageName} not found in {relativePaths}")

    def readNiftiImage(self, filePath):
        # Read Nifti image
        try:
            niftiImage = nib.load(filePath).get_fdata()
            return niftiImage, nib.load(filePath).affine
        except Exception as e:
            print(f"Error reading NIFTI image from {filePath}: {str(e)}")

    def plot_original_images(self, vec_img, case, counter,  axes, title, slice=20):
        # Loop through each case and its images, and plot them in the subplots
        for j in range(len(vec_img)):  # 3 images per case
            ax = axes[j, counter]
            ax.set_xticks([])
            ax.set_yticks([])
            if j >= 1:
                ax.imshow(vec_img[j][:, :, slice], cmap='viridis')
            else:
                ax.imshow(vec_img[j][:, :, slice], cmap='gray')

            if j == 0:
                ax.set_title(f'Case {case}')

            if counter == 0:
                ax.set_ylabel(title[j])

    @staticmethod
    def getRegistrationSortKey(filePath, fallbackSortValue = 99):
        # first sorted by registration type, than by filename
        primarySortOrder = {
            'rigid': 1,
            'affine': 2,
            'bspline': 3
        }
        primarySortValue = fallbackSortValue 
        fileName = os.path.basename(filePath)  
        secondarySortValue = fileName
        
        for keyword, orderValue in primarySortOrder.items():
            if keyword in fileName:
                primarySortValue = orderValue
                break
        return (primarySortValue, secondarySortValue)

    @staticmethod
    def ensureFolderExists(folderPath):
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    
    @staticmethod
    def splitNameFromExtension(filePath):
        name = pathlib.Path(filePath).stem
        extension = "".join([s for s in pathlib.Path(filePath).suffixes if not " " in s])
        return name, extension




