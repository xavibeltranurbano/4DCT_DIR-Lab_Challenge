# # -----------------------------------------------------------------------------
# # Postprocessing of the segmentation
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------

import numpy as np
from scipy.ndimage import label
import cv2

class Postprocessing:
    def __init__(self):
        pass
    
    def postprocessing(self, mask):
        largestComponents =  self.findThreeLargestComponents(mask)
        remainingComponents = self.removeComponentsTouchingEdges(largestComponents)
        remainingComponents = self.findLung(remainingComponents)
        return self.combineMasks(remainingComponents)

    @staticmethod
    def findThreeLargestComponents(binaryMask):
        # Label all connected components
        labeledArray, numFeatures = label(binaryMask)
        componentSizes = np.zeros(numFeatures + 1, dtype=int)
        for i in range(1, numFeatures + 1):
            componentSizes[i] = np.sum(labeledArray == i)

        largestComponents = np.argsort(componentSizes)[-3:]
        largestComponentMasks = []
        for component in largestComponents:
            if componentSizes[component] > 0:  
                mask = (labeledArray == component).astype(int)
                largestComponentMasks.append(mask)

        return largestComponentMasks

    @staticmethod
    def removeComponentsTouchingEdges(largestComponents):
        remainingComponents = []
        for component in largestComponents:
            side1 = component[-1, :, :]    
            side2 = component[:, -1, :]  
            side3 = component[:, :, -1]
            side4 = component[0, :, :]    
            side5 = component[:, 0, :]  
            side6 = component[:, :, 0]

            if not np.any(side1) and not np.any(side2) and not np.any(side3) and not np.any(side4) and not np.any(side5) and not np.any(side6) :
                remainingComponents.append(component)

        return remainingComponents

    @staticmethod
    def findLung(masks):
        masks = masks[::-1]
        sizePerMask = [] 
        for mask in masks:
            size = np.sum(mask)
            sizePerMask.append(size)

        # lung (biggest) + lung (2nd biggest) or lung
        if len(masks) >=2 and np.isclose(sizePerMask[0], sizePerMask[1], rtol=0.5):
            return [masks[0], masks[1]]
        else:
            return [masks[0]]

    @staticmethod
    def combineMasks(maskList):
        combinedMask = np.zeros_like(maskList[0], dtype=int)
        for mask in maskList:
            combinedMask = np.logical_or(combinedMask, mask).astype(int)
        return combinedMask

