# # -----------------------------------------------------------------------------
# # Main File to reproduce segmentations of the COPDgene dataset
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------

import SimpleITK as sitk
import os
from lungSegmentation import LungSegmentation

def segmentAndSaveImage(originalImagePath, segmentedImagePath, segmentedMaskPath):
    # Read the original image
    originalImage = sitk.ReadImage(originalImagePath)
    originalData = sitk.GetArrayFromImage(originalImage)

    # Segment the image 
    lungSeg = LungSegmentation(originalImagePath)
    predictedMask = lungSeg.segmentLung()
    segmentedData = predictedMask * originalData

    # Create a new SimpleITK image for the segmented data
    segmentedMask = sitk.GetImageFromArray(predictedMask)
    segmentedImage = sitk.GetImageFromArray(segmentedData)

    # Copy metadata from the original image to the segmented image
    for key in originalImage.GetMetaDataKeys():
        segmentedImage.SetMetaData(key, originalImage.GetMetaData(key))
        segmentedMask.SetMetaData(key, originalImage.GetMetaData(key))

    # Set the same spacing, origin, and direction as the original image
    for key in originalImage.GetMetaDataKeys():
        value = originalImage.GetMetaData(key)
        segmentedImage.SetMetaData(key, value)
        segmentedMask.SetMetaData(key, value)

    # Save the segmented image with the original metadata
    os.makedirs(os.path.dirname(segmentedImagePath), exist_ok=True)
    os.makedirs(os.path.dirname(segmentedMaskPath), exist_ok=True)
    sitk.WriteImage(segmentedImage, segmentedImagePath)
    sitk.WriteImage(segmentedMask, segmentedMaskPath)


if __name__ == "__main__":
    # Example usage on the COPDgene dataset
    datasetDirectory = "data"

    for i in range(1,5):
        for status in ["i", "e"]:
            originalImagePath = os.path.join(datasetDirectory, f"copd{i}/copd{i}_{status}BHCT.nii")
            segmentedImagePath = os.path.join(datasetDirectory, f"copd{i}/segmentations/copd{i}_{status}BHCT_segmented.nii")
            segmentedMaskPath = os.path.join(datasetDirectory, f"copd{i}/segmentations/copd{i}_{status}BHCT_mask.nii")

            print(originalImagePath)
            segmentAndSaveImage(originalImagePath, segmentedImagePath, segmentedMaskPath)
    