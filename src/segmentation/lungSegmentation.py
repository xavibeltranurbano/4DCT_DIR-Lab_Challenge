# # -----------------------------------------------------------------------------
# # Preprocessing of the segmentation
# # Author: Frederik Hartmann, Yusuf Baran Tanrıverdi and Xavier Beltran Urbano
# # Date Created: 01-03-2023
# # -----------------------------------------------------------------------------

from preprocessing import Preprocessing
from postprocessing import Postprocessing
from ploting import Ploting

import numpy as np
import cv2
import SimpleITK as sitk

class LungSegmentation:
    JACCARD_THRESHOLD = 0.05
    MAX_NUMBER_OF_CONTOURS_TO_BE_TRACKED = 6
    preprocessing = Preprocessing()
    postprocessing = Postprocessing()
    plot = Ploting()

    def __init__(self, scanPath):
        self.scanMeta = self.readScanMetaFrom(scanPath)
        self.scan = self.metaToScan()
        self.scanDimensions = self.scan.shape


    @staticmethod
    def readScanMetaFrom(scanPath):
        return sitk.ReadImage(scanPath)

    def metaToScan(self):
        return sitk.GetArrayFromImage(self.scanMeta)
    
    def segmentLung(self):
        coarseMask = self.preprocessing.createCoarseLungMaskOf(self.scan)
        coarseScan = coarseMask*self.scan
        fineMask = self.refine(coarseScan)
        postprocessedMask = self.postprocessing.postprocessing(fineMask)
        return postprocessedMask
    
    def refine(self, coarseScan):
        coarseScan[coarseScan==0] = coarseScan.max()
        HounsfieldUnitRange = (100, 700)
        clippedScan = self.preprocessing.clipScanToHounsfieldUnitRange(coarseScan, HounsfieldUnitRange)
        contoursForEachAxialSlice = self.findContoursForEachAxialSliceOf(clippedScan)
        
        fineMask = self.createFineMaskFrom(contoursForEachAxialSlice)
        return fineMask

    def findContoursForEachAxialSliceOf(self, clippedScan):
        numberOfAxialSlices = clippedScan.shape[0]
        contoursForEachAxialSlice = [None] * numberOfAxialSlices
        for i in range(0, numberOfAxialSlices):
            axialSlice = clippedScan[i]
            sliceContours = self.findContoursOf(axialSlice)
            contoursForEachAxialSlice[i] = sliceContours
            centerIndex = int(len(contoursForEachAxialSlice)/2)

        return contoursForEachAxialSlice
    
    def findContoursOf(self, axialSlice):
        preparedSlice = self.prepare(axialSlice)
        contours,_ = cv2.findContours(preparedSlice, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contoursSortedByLength = sorted(contours, key=lambda contour:len(contour), reverse=True)
        reducedNumberOfContours = self.reduceNumberOf(contoursSortedByLength)
        return reducedNumberOfContours

    def prepare(self, axialSlice):
        denoisedAxialSlice = cv2.medianBlur(axialSlice, ksize=5)
        binarizedAxialSlice = self.binarize(denoisedAxialSlice)
        return binarizedAxialSlice

    @staticmethod
    def binarize(axialSlice):
        # invert threshold so lung is white(foreground) and background is black
        _, binarizedSlice = cv2.threshold(axialSlice, thresh=axialSlice.max()-1, maxval=1, type=cv2.THRESH_BINARY_INV)
        return binarizedSlice.astype("uint8")
    
    def reduceNumberOf(self, contoursSortedByLength):
        if len(contoursSortedByLength) > self.MAX_NUMBER_OF_CONTOURS_TO_BE_TRACKED:
            return contoursSortedByLength[:self.MAX_NUMBER_OF_CONTOURS_TO_BE_TRACKED]
        else:
            return contoursSortedByLength
    
    def createFineMaskFrom(self, contoursForEachAxialSlice):
        refinedBottomMask = self.refineMaskBytrackingLungContours(contoursForEachAxialSlice, direction="centerToTop")
        refinedTopMask = self.refineMaskBytrackingLungContours(contoursForEachAxialSlice, direction="centerToBottom")
        refinedMask = self.addMasks(refinedBottomMask,refinedTopMask)
        return refinedMask
    
    def refineMaskBytrackingLungContours(self, contoursForEachAxialSlice, direction):
        startIndex = self.getStartIndex(contoursForEachAxialSlice, direction)
        finalIndex = self.getFinalIndex(contoursForEachAxialSlice, direction)
        stepDirection = self.stepDirectionToInteger(direction)
        refinedMask = np.zeros(self.scanDimensions)

        centerContours = contoursForEachAxialSlice[startIndex]
        previousMasks = self.getCandidateMasksFrom(centerContours)
        for i in range(startIndex, finalIndex, stepDirection):
            CurrentContours = contoursForEachAxialSlice[i]
            lungMasks = self.comparePreviousMasksToCurrentContours(previousMasks,CurrentContours)
            lungMask = self.createSingleMaskFrom(lungMasks)
            refinedMask[i] = lungMask
            previousMasks = lungMasks

        return refinedMask

    @staticmethod
    def getStartIndex(contoursForEachAxialSlice, direction):
        centerIndex = int(len(contoursForEachAxialSlice)/2)
        if direction == "centerToTop":
            centerIndex += 1
        return centerIndex
        
    @staticmethod
    def getFinalIndex(contoursForEachAxialSlice, direction):
        if direction == "centerToTop":
            finalIndex = len(contoursForEachAxialSlice)            
        else:
            finalIndex = 0
        return finalIndex
    
    @staticmethod
    def stepDirectionToInteger(direction):
        if direction == "centerToTop":
            stepDirection = 1
        else:
            stepDirection = -1
        return stepDirection 
    
    def getCandidateMasksFrom(self, CurrentContours):
        numberOfContours = len(CurrentContours)
        candidateMasks = [None] * numberOfContours
        sliceDimensions = self.scanDimensions[1:3]

        for i in range(0,numberOfContours):
            candidateMask = np.zeros(sliceDimensions, dtype="uint8")
            cv2.drawContours(candidateMask, [CurrentContours[i]], contourIdx=-1, color=1, thickness=cv2.FILLED)
            candidateMasks[i] = candidateMask

        return candidateMasks

    def comparePreviousMasksToCurrentContours(self, previousMasks,CurrentContours):
        candidateMasks = self.getCandidateMasksFrom(CurrentContours)
        lungMasks = []

        for mask in candidateMasks:
            for prevMask in previousMasks:
                if self.isLungMask(mask, prevMask):
                    lungMasks.append(mask)
                    break # breaks inner loop

        return lungMasks

    def isLungMask(self,mask, prevMask):
        jaccardScore = self.computeJaccardScore(mask, prevMask)
        currentSize = cv2.countNonZero(mask)
        prevSize = cv2.countNonZero(prevMask)

        if self.isMaskOverlapping(jaccardScore):
            return True
        elif self.isMaskSplittedIntoTwoMasks(jaccardScore, currentSize, prevSize):
            return True
        elif self.isMaskMergedFromTwoMasks(jaccardScore, currentSize, prevSize):
            return True
        else:
            return False
    
    @staticmethod
    def computeJaccardScore(mask, prevMask):
        intersection = cv2.countNonZero(cv2.bitwise_and(mask,prevMask))
        union = cv2.countNonZero(cv2.bitwise_or(mask,prevMask))
        if union != 0:
            return intersection/union
        else:
            return 1.0

    def isMaskOverlapping(self, jaccardScore):
        if jaccardScore > self.JACCARD_THRESHOLD:
            return True
        else:
            return False
    
    def isMaskSplittedIntoTwoMasks(self, jaccardScore, currentSize, prevSize):
        if jaccardScore > self.JACCARD_THRESHOLD/3 and np.isclose(2*currentSize, prevSize, rtol=0.3):
            return True
        else:
            return False
    
    def isMaskMergedFromTwoMasks(self, jaccardScore, currentSize, prevSize):
        if jaccardScore > self.JACCARD_THRESHOLD/3 and np.isclose(currentSize, 2*prevSize, rtol=0.3):
            return True
        else:
            return False

    def createSingleMaskFrom(self, axialLungMasks):
        singleMask = np.zeros(self.scanDimensions[1:3])
        for mask in axialLungMasks:
            singleMask += mask
        return singleMask

    @staticmethod
    def addMasks(refinedBottomMask,refinedTopMask):
        mask = np.zeros(refinedBottomMask.shape, dtype="uint8")
        numberOfAxialSlices = refinedBottomMask.shape[0]
        for i in range(0,numberOfAxialSlices):
            mask[i] = refinedBottomMask[i] + refinedTopMask[i]
        return mask



