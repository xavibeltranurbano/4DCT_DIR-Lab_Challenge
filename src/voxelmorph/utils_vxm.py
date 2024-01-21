# # -----------------------------------------------------------------------------
# # Utility File For Voxelmorph
# # Author: Xavier Beltran Urbano and Frederik Hartmann
# # Date Created: 12-12-2023
# # -----------------------------------------------------------------------------


import itk
import os
import pathlib
import nibabel as nib
import sys
import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import voxelmorph as vxm
from dataGenerator import CTDataGenerator
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('/notebooks/'))
from evaluation import Evaluation


class Utils:
    def __init__(self, imgPath):
        self.imgPath=imgPath
        
    def zeroPadding(self,img,nSlices=128):
        # Zero padding at the end of the image
        single_slice=np.zeros((512,512,1))
        for slice in range(nSlices-img.shape[-1]):
            img=np.concatenate([img,single_slice],axis=-1)
        return img

    def resizeImage(self,img):
        # Resize image
        resized_img=np.zeros((256,256,128))
        for slice in range(126):
            resized_img[:,:,slice]=cv2.resize(img[:,:,slice],(256,256))
        return resized_img

    def load_data(self):
        # Load data into a vector
        file_names = [file_name for file_name in os.listdir(self.imgPath) if not "DS_Store" in file_name]
        all_images_ex = all_images_in= []
        for type_scan in ['iBHCT', 'eBHCT']:
            for name in file_names:
                # Construct the full path to the image
                image_path = os.path.join(self.imgPath, name, f"segmentations/{name}_{type_scan}_segmented.nii")
                # Load and append the image data
                nifti_image = nib.load(image_path)
                image_data = nifti_image.get_fdata()
                if image_data.shape[-1]!=126:
                    image_data=self.zeroPadding(image_data)
                image_data=self.resizeImage(image_data) # resize to 256x256 to faster computation
                if type_scan =='iBHCT': all_images_in.append(image_data)
                else:  all_images_ex.append(image_data)
        return all_images_in,all_images_ex                                                      
        
    def save_training(self,hist, loss_name='loss', filename='training_history.png'):
        # Simple function to plot training history.
        plt.figure()
        plt.plot(hist.epoch, hist.history[loss_name], '.-')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # Save the plot to a file
        plt.savefig(filename)
        plt.close()  # Close the figure to free up memory

    def get_landmarks(self, img_number):
        # Read the landmarks from csv files
        landmarks_path_exhale = f'{self.imgPath}/copd{img_number}/copd{img_number}_300_eBH_xyz_r1.txt'
        landmarks_exhale = pd.read_csv(landmarks_path_exhale, header=None, sep='\s+', engine='python').values
        landmarks_path_inhale =  f'{self.imgPath}/copd{img_number}/copd{img_number}_300_iBH_xyz_r1.txt'
        landmarks_inhale = pd.read_csv(landmarks_path_inhale, header=None, sep='\s+', engine='python').values
        return landmarks_exhale,landmarks_inhale
    
    def register_landmarks(self, processed_landmarks,def_field,img_shape):
        # Apply registration to landmarks
        data = [tf.convert_to_tensor(f, dtype=tf.float32) for f in [processed_landmarks[np.newaxis,...], def_field[np.newaxis,...]]]
        annotations_warped = vxm.utils.point_spatial_transformer(data)[0, ...].numpy()
        registration_landmarks=np.multiply(annotations_warped, [2,2,img_shape/128])
        # Round the values and then convert to integers
        registration_landmarks_int = np.round(registration_landmarks).astype(int)
        return registration_landmarks_int
        
    def compute_Metrics(self,vxm_model):
        # Compute metrics with the trained model
        print("Starting to predict the images...")
        vec_slices=[121,102,126,126]
        inhalation_images,exhalation_images = self.load_data()
        val_generator = CTDataGenerator(inhalation_images, exhalation_images, batch_size=1)
        evaluation=Evaluation()
        all_tre=[]
        print(" ")
        for i in range(1,len(vec_slices)+1):
            # Take input samples
            in_sample, _ =val_generator[i]
            # Predict the samples
            val_pred = vxm_model.predict(in_sample,verbose=0)
            prediction=val_pred[0]
            # Save the registered image
            nifti_image = nib.Nifti1Image(prediction[0,...,0], affine=np.eye(4))
            nib.save(nifti_image, f'/notebooks/voxelmorph/results/registered_image{i}.nii')
            # Compute TRE
            def_field=val_pred[1].squeeze()
            landmark_ex, landmark_in= self.get_landmarks(i)
            factors = [0.5, 0.5, 128/vec_slices[i-1]]
            processed_landmarks = np.multiply(landmark_in, factors)
            registererd_landmarks=self.register_landmarks(processed_landmarks,def_field,vec_slices[i-1])
            tre=evaluation.targetRegistrationError(registererd_landmarks,landmark_ex)
            print(f"TRE Image {i}: {tre}")
            all_tre.append(tre)
        print(f"-------------------")
        print(f"Mean TRE: {np.mean(np.asarray(all_tre))}")
