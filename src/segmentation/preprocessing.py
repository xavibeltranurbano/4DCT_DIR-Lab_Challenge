# # -----------------------------------------------------------------------------
# # Preprocessing of the segmentation
# # Author: Frederik Hartmann, Yusuf Baran Tanrıverdi and Xavier Beltran Urbano
# # Date Created: 01-03-2023
# # -----------------------------------------------------------------------------

import numpy as np
import cv2
from ploting import Ploting

class Preprocessing:
    plot = Ploting()
    minDerivative = 1e10
    maskSizes = None


    def __init__(self):
        pass

    def createCoarseLungMaskOf(self,scan):
        HounsfieldUnitRange = (100, 700)
        clippedScan = self.clipScanToHounsfieldUnitRange(scan, HounsfieldUnitRange)
        mask = self.createMaskOf(clippedScan)
        mask, nBrokenSlices = self.replaceBrokenSlice(mask)

        i = 1
        while nBrokenSlices > 0 and i < mask.shape[1]:
            mask, nBrokenSlices = self.replaceBrokenSlice(mask)
            i+=1

        numberOfAxialSlices = mask.shape[0]
        for i in range(0,numberOfAxialSlices):
            axialMask = mask[i]
            inverted = (np.logical_not(axialMask)).astype("uint8")
            sliceMask = self.createMaskByFillingHolesOf(inverted)
            mask[i] = sliceMask.astype("int16") 
        return mask
    
    @staticmethod
    def clipScanToHounsfieldUnitRange(scan,HounsfieldUnitRange):
        HU_min = HounsfieldUnitRange[0]
        HU_max = HounsfieldUnitRange[1]
        return np.clip(scan,a_min=HU_min, a_max=HU_max) # 16-bit signed!
    
    def createMaskOf(self, clippedScan):
        return self.createMaskForEachSliceOf(clippedScan)

    def createMaskForEachSliceOf(self, clippedScan):
        mask = np.zeros(clippedScan.shape, dtype="int16")
        numberOfSagittalSlices = clippedScan.shape[1]
        for i in range(0,numberOfSagittalSlices):
            sagittalSlice = clippedScan[:,:,i]
            sliceMask = self.createMaskFrom(sagittalSlice)
            mask[:,:,i] = sliceMask.astype("int16")
        return mask
    
    def createMaskFrom(self,SagittalSlice):
        denoisedSagittalSlice = cv2.medianBlur(SagittalSlice,ksize=5)
        binarizedSagittalSlice = self.binarize(denoisedSagittalSlice)
        
        # Open the table to allow for creation of a uniform background.
        sliceWithOpenTable = self.openTableOf(binarizedSagittalSlice)

        # the implemented creation of a mask will increase the mask into the previosuly filled background.
        # To reduce the effect, the background mask is combined with the mask.
        SagittalSliceWithUniformBackground, backgroundMask = self.createUniformBackgroundOf(binarizedSagittalSlice)
        mask = self.createMaskByFillingHolesOf(SagittalSliceWithUniformBackground)
        combinedMask = self.combineMasks(mask, backgroundMask)
        return combinedMask

    
    @staticmethod
    def binarize(sagittalSlice):
        _, binarizedSlice = cv2.threshold(sagittalSlice, thresh=sagittalSlice.max()-1, maxval=1, type=cv2.THRESH_BINARY)
        return binarizedSlice.astype("uint8")

    @staticmethod
    def openTableOf(binarizedSagittalSlice):
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(25,1))
        topRowOpen = cv2.morphologyEx(binarizedSagittalSlice[:1], cv2.MORPH_OPEN, kernel, iterations=1)
        binarizedSagittalSlice[:1] = topRowOpen
        return binarizedSagittalSlice
    
    @staticmethod
    def createUniformBackgroundOf(binarizedSagittalSlice):
        #safety feature
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(20,1))
        binarizedSagittalSlice = cv2.morphologyEx(binarizedSagittalSlice, cv2.MORPH_CLOSE, kernel, iterations=1)

        # floodfill the slice from the left -to fill the bodybackground-  and the right - to fill the table area.
        h, w = binarizedSagittalSlice.shape[:2]
        backgroundMask = np.zeros((h+2,w+2),dtype="uint8")
        cv2.line(binarizedSagittalSlice, (0,0),(0,h-1),0,thickness=1) 
        cv2.line(binarizedSagittalSlice, (w-1,0),(w-1,h-1),0,thickness=1) 
        cv2.floodFill(binarizedSagittalSlice, backgroundMask, (0,0),1,flags=4) 
        cv2.floodFill(binarizedSagittalSlice, backgroundMask, (w-1,0),1,flags=4)
        backgroundMask = np.logical_not(backgroundMask[1:-1,1:-1]).astype("uint8") 
        return binarizedSagittalSlice, backgroundMask
    

    @staticmethod
    def createMaskByFillingHolesOf(BinarySliceWithUniformBackground):
        inverted = (np.logical_not(BinarySliceWithUniformBackground)).astype("uint8")
        filled = cv2.boxFilter(inverted, ddepth=-1, ksize=(30,30), normalize=False)
        _, binarized = cv2.threshold(filled, thresh=1, maxval=1, type=cv2.THRESH_BINARY)
        return binarized

    @staticmethod
    def combineMasks(filledMask, backgroundMask):
        return cv2.bitwise_and(filledMask, backgroundMask)

    def replaceBrokenSlice(self, mask):
        brokenSliceIndices = self.findBrokenSlices(mask)
        if len(brokenSliceIndices) == 0:
            return mask, 0
        
        brokenSliceIndex = brokenSliceIndices[0]
        workingSliceIndex = brokenSliceIndex - 1
        mask[:,:,brokenSliceIndex] = mask[:,:,workingSliceIndex]
        self.maskSizes[brokenSliceIndex] = cv2.countNonZero(mask[:,:,brokenSliceIndex])
        return mask, len(brokenSliceIndices) - 1

    def findBrokenSlices(self, mask):
        if not self.maskSizes:
            self.maskSizes = self.calculateMaskSizePerSagittalSlice(mask)
        normalizedDerivative = self.calcDerivative()
        brokenIndices = (np.argwhere(normalizedDerivative > 0.1)).flatten()
        brokenIndices += 1 # adjust for gradient
        return brokenIndices

    @staticmethod
    def calculateMaskSizePerSagittalSlice(mask):
        numberOfSagittalSlices = mask.shape[1]
        maskSizes = []
        for i in range(0,numberOfSagittalSlices):
            currentSliceMask = mask[:,:,i]
            size = cv2.countNonZero(currentSliceMask)
            maskSizes.append(size)
        return maskSizes

    def calcDerivative(self):
        firstDerivative = np.gradient(self.maskSizes, edge_order=1)
        minimum = np.min(firstDerivative)
        if minimum < self.minDerivative:
            self.minDerivative = minimum
        negativeDerivative = np.clip(firstDerivative, a_min=self.minDerivative, a_max=0)
        normalizedDerivative = negativeDerivative / self.minDerivative
        return normalizedDerivative


