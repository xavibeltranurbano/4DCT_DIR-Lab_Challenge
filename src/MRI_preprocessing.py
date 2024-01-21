# -----------------------------------------------------------------------------
# Preprocessing  file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------


import os
import nibabel as nib
import numpy as np
import random
import tensorflow as tf
from dataAugmentation import AugmentedDataGenerator

class Preprocessing():
    def __init__(self,data_aug,norm_intensity):
        self.dataAug = data_aug
        self.normIntensity = norm_intensity

    def readImage(self, image_path):
        img = nib.load(image_path)
        img = img.get_fdata()
        return img

    def dataAugmentation(self, images, masks):
        augmented_img=[]
        augmented_mask=[]
        for img, mask in zip(images,masks):
            rand_flip1 = np.random.randint(0, 2)
            rand_flip2 = np.random.randint(0, 2)
            if rand_flip1 == 1:
                img = np.flip(img, 0)
                mask = np.flip(mask, 0)
            if rand_flip2 == 1:
                img = np.flip(img, 1)
                mask = np.flip(mask, 1)
            augmented_img.append(img)
            augmented_mask.append(mask)
        return augmented_img, augmented_mask
        
    def zscore(self,img):
        lower_percentile = np.percentile(img, 25)
        upper_percentile = np.percentile(img, 75)
        filtered_array = img[(img >= lower_percentile) & (img <= upper_percentile)]
        mean = np.mean(filtered_array)
        std = np.std(filtered_array)
        z_scores = (img - mean) / std
        return z_scores
    
    def normalizeIntensity(self,img):
        img[img>0]=self.zscore(img[img>0])
        return img

    def pad_image_center_sagital(self,image):
        total_padding = 256 - image.shape[0]
        top_padding = total_padding // 2
        bottom_padding = total_padding - top_padding
        padded_image = np.pad(image, ((top_padding, bottom_padding), (0, 0)), mode='constant', constant_values=0)
        return padded_image
    
    def pad_image_center_axial(self,image):
        total_padding = 256 - image.shape[1]
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        padded_image = np.pad(image, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=0)
        return padded_image
    
    def selectSlices(self, img, gt, numberSlices):
        allSlices_y = []
        allSlices_x = []
        csf_slices_x = []
        csf_slices_y = []
        # Iterate through each slice
        for slice_idx in range(gt.shape[1]):
            #slice_y = gt[:, :, slice_idx, 0] # Axial cases
            #slice_x = img[:, :, slice_idx, 0] # Axial cases
            slice_y = gt[:, slice_idx, :, 0]
            slice_x = img[:, slice_idx, :, 0]
            ## Check if the slice is not entirely background
            if not np.all(slice_y == 0):
                #slice_y=self.pad_image_center_axial(slice_y) # Axial cases
                #slice_x=self.pad_image_center_axial(slice_x) # Axial cases
                # Check if it contains CSF label
                if np.any(slice_y == 1):
                    csf_slices_y.append(slice_y)
                    csf_slices_x.append(slice_x)
                else:
                    allSlices_y.append(slice_y)
                    allSlices_x.append(slice_x)

        # If the number of CSF slices is exactly what we need
        if len(csf_slices_y) == numberSlices:
            return csf_slices_x, csf_slices_y
        # If there are more CSF slices than needed
        elif len(csf_slices_y) > numberSlices:
            selected_indices = random.sample(range(len(csf_slices_y)), numberSlices)
            selected_slices_y = [csf_slices_y[i] for i in selected_indices]
            selected_slices_x = [csf_slices_x[i] for i in selected_indices]
            return selected_slices_x, selected_slices_y
        # If there are fewer CSF slices than needed
        else:
            needed_slices = numberSlices - len(csf_slices_y)
            selected_indices = random.sample(range(len(allSlices_y)), needed_slices)
            selected_slices_y = [allSlices_y[i] for i in selected_indices]
            selected_slices_x = [allSlices_x[i] for i in selected_indices]
            csf_slices_x.extend(selected_slices_x)
            csf_slices_y.extend(selected_slices_y)
            return csf_slices_x, csf_slices_y
            
    def to_one_hot(self,y, num_classes):
        'Creates a one-hot-encoded tensor of y_true with num_classes'
        y = tf.cast(y, dtype='int32')
        y = tf.one_hot(y, depth=num_classes)
        return y
    
    def process_case(self,ID,path, numberSlices, nClasses):
        path_img=os.path.join(path, ID,ID+".nii.gz")
        path_gt=os.path.join(path, ID,ID+"_seg.nii.gz")
        # Read images
        
        img=self.readImage(path_img)
        gt=self.readImage(path_gt)
        
        # Normalize intensities
        if self.normIntensity:
            img=self.normalizeIntensity(img) # We do not normalise the background
        
        # Select slices with some criteria
        if numberSlices>0:
            img,gt= self.selectSlices(img, gt, numberSlices)
        else: #val data
            img=np.transpose(img[...,0],[1,0,2])
            gt=np.transpose(gt[:,:,:,0],[1,0,2])

        # Data augmentation
        if self.dataAug:
            img,gt=self.dataAugmentation(img,gt)
        
        # Transform the GT to one hot encoded
        gt=self.to_one_hot(gt, nClasses)
        img=np.asarray(img)
        return img, gt