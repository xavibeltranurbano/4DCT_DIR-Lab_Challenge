# -----------------------------------------------------------------------------
# Predict  file
# Author: Xavier Beltran Urbano and Frederik Hartmann
# Date Created: 12-01-2024
# -----------------------------------------------------------------------------

import os
import numpy as np
from metrics import *
import nibabel as nib
from keras.models import load_model
import matplotlib.pyplot as plt
from MRI_preprocessing import Preprocessing 
from scipy.spatial.distance import directed_hausdorff


class Prediction:
    def __init__(self, models_path, val_data_dir, preprocessing, plane_types):
        self.models = self.load_pretrained_model(models_path)
        self.val_data_dir = val_data_dir
        self.test_data_dir = '/notebooks/data/Test_Set'
        self.preprocessing = preprocessing
        self.plane_types = plane_types
        self.dice_scores = {'csf': [], 'gm': [], 'wm': []}
        self.hausdorff_distances = {'csf': [], 'gm': [], 'wm': []}

    @staticmethod
    def dice_coefficient(y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1e-6) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)
    
    @staticmethod
    def hausdorff_distance(y_true, y_pred):
        return max(directed_hausdorff(y_true, y_pred)[0], directed_hausdorff(y_pred, y_true)[0])


    @staticmethod
    def load_pretrained_model(models_path):
        all_models = []
        different_config=[]
        for path in models_path:
            if "SegResNet" in path:
                all_models.append(load_model(path, custom_objects={
                    'categoricalCrossentropy': categoricalCrossentropy,
                    'mean_dice_coef': mean_dice_coef,
                    'csf': csf,
                    'gm': gm,
                    'wm': wm,
                    'bckgrd': bckgrd }))
            else:
                all_models.append(load_model(path, custom_objects={
                    'categoricalCrossentropy': categoricalCrossentropy,
                    'mean_dice_coef': mean_dice_coef}))
                
        return all_models

    @staticmethod
    def load_data(img_data_dir, preprocessing, set_type):
        if set_type=='val':
            img_ids = ['IBSR_11', 'IBSR_12', 'IBSR_13', 'IBSR_14', 'IBSR_17']
        else:
            img_ids=['IBSR_02','IBSR_10','IBSR_15']
        img_data = []
        for id in img_ids:
            image_path = os.path.join(img_data_dir, id, id + '.nii.gz')
            image = preprocessing.readImage(image_path)
            image = np.transpose(image, [1, 0, 2, 3])
            if set_type=='val':
                mask_path = os.path.join(img_data_dir, id, id + '_seg.nii.gz')
                mask = preprocessing.readImage(mask_path)
                mask = np.transpose(mask, [1, 0, 2, 3])
                img_data.append((id, image, mask))
            else: img_data.append((id, image))
        return img_data

    @staticmethod
    def predict_models(models, plane_type, img):
        all_predictions = []
        for model, plane in zip(models, plane_type):
            if plane == 'coronal':
                pred_mask = model.predict(img, verbose=0)
            elif plane == 'axial':
                pred_mask = Prediction.predict_axial(model, img)
            else: #3D
                img_new=np.expand_dims(np.transpose(img,[1,0,2,3]),axis=0)
                pred_mask = model.predict(img_new, verbose=0)
                pred_mask=np.transpose(pred_mask[0,...],[1,0,2,3])
            all_predictions.append(pred_mask)
        return all_predictions

    @staticmethod
    def weighted_probabilities(all_preds, weight_models):
        weighted_probs = []
        for pred, weight in zip(all_preds, weight_models):
            weighted_probs.append(pred * weight)
        return np.array(weighted_probs)

    @staticmethod
    def ensemble_models(models, img, plane_type, method='majority', weights=None):
        all_preds = Prediction.predict_models(models, plane_type, img)

        if method == 'majority':
            all_preds_labels = [np.argmax(pred, axis=-1) for pred in all_preds]
            ensemble_pred = np.array(all_preds_labels).astype(np.int32)
            ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(models)).argmax(), axis=0,
                                                arr=ensemble_pred)
            ensemble_pred = np.expand_dims(ensemble_pred, axis=-1)

        elif method == 'mean':
            ensemble_pred = np.mean(np.array(all_preds), axis=0)
            ensemble_pred = np.argmax(ensemble_pred, axis=-1)

        elif method == 'maximum':
            ensemble_pred = np.max(np.array(all_preds), axis=0)
            ensemble_pred = np.argmax(ensemble_pred, axis=-1)

        return ensemble_pred

    @staticmethod
    def pad_image_center_axial(image):
        total_padding = 256 - image.shape[1]
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        padded_image = np.pad(image, ((0, 0), (left_padding, right_padding)), mode='constant', constant_values=0)
        return padded_image

    @staticmethod
    def remove_padding_axial(padded_image):
        all_channels=[]
        for channel in range(padded_image.shape[-1]):
            one_channel_image = padded_image[..., channel]
            original_width = 128
            left_padding = (one_channel_image.shape[1] - original_width) // 2
            start_idx = left_padding
            end_idx = start_idx + original_width
            all_channels.append(one_channel_image[:, start_idx:end_idx])
        return np.stack(all_channels, axis=-1)

    @staticmethod
    def predict_axial(model, img):
        nifti_data = np.transpose(img, [1, 0, 2, 3])
        predicted_slices = []
        for i in range(nifti_data.shape[2]):
            slice_2d = Prediction.pad_image_center_axial(nifti_data[:, :, i, 0])
            preprocessed_slice = np.expand_dims(slice_2d, axis=0)
            prediction_axial = model.predict(preprocessed_slice, verbose=0)
            prediction_axial = np.squeeze(prediction_axial)
            prediction_axial = Prediction.remove_padding_axial(prediction_axial)
            predicted_slices.append(prediction_axial)

        predicted_3d = np.stack(predicted_slices, axis=2)
        predicted_3d = np.transpose(predicted_3d, [1, 0, 2, 3])
        return predicted_3d

    def predict_and_evaluate_case(self, case_id, img, mask,set_type):
        print(f"Evaluating case: {case_id}")

        img = self.preprocessing.normalizeIntensity(img)
        if len(self.models) == 1:
            if self.plane_types[0] == 'coronal':
                pred_mask = self.models[0].predict(img, verbose=0)
                pred_mask_discrete = np.argmax(pred_mask, axis=-1)
            elif self.plane_types[0] == 'axial':
                pred_mask_discrete = Prediction.predict_axial(self.models[0], img)
                pred_mask_discrete = np.argmax(pred_mask_discrete, axis=-1)
            else: #3D
                img=np.expand_dims(np.transpose(img,[1,0,2,3]),axis=0)
                mask=np.expand_dims(np.transpose(mask,[1,0,2,3]),axis=0)
                pred_mask = self.models[0].predict(img, verbose=0)
                pred_mask_discrete = np.argmax(pred_mask[0,...], axis=-1)
        else:
            pred_mask_discrete = Prediction.ensemble_models(self.models, img, self.plane_types, method='mean')
        
        if set_type=='val':
            for tissue_type, label in [('csf', 1), ('gm', 2), ('wm', 3)]:
                true_mask = (mask == label)
                predicted_mask = (pred_mask_discrete == label)

                dice_score = Prediction.dice_coefficient(true_mask, predicted_mask)
                self.dice_scores[tissue_type].append(dice_score)
                if true_mask.ndim>3:
                    true_mask=true_mask[...,0]
                if predicted_mask.ndim>3:
                    predicted_mask=predicted_mask[...,0]
                hausdorff_dist = Prediction.hausdorff_distance(np.argwhere(true_mask), np.argwhere(predicted_mask))
                # Store the Hausdorff distance in a similar way to dice_scores
                self.hausdorff_distances[tissue_type].append(hausdorff_dist)
                print(f"Tissue: {tissue_type} - Dice Score: {dice_score}, Hausdorff Distance: {hausdorff_dist}")
            print(" ")
            return pred_mask_discrete
        else: return pred_mask_discrete

    def predict_test(self, output_folder):
        test_data = self.load_data(self.test_data_dir, self.preprocessing,'test')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for case_id, img in test_data:
            pred_mask_discrete = self.predict_and_evaluate_case(case_id, img, [],'test')
            pred_mask_discrete = np.transpose(pred_mask_discrete, [1, 0, 2])
            pred_nifti = nib.Nifti1Image(pred_mask_discrete.astype(np.uint16), affine=np.eye(4))
            nib.save(pred_nifti, os.path.join(output_folder, case_id + '_pred.nii.gz'))
        print("----------------------------------------------")
        print("\nPrediction has been done.")


    def run_evaluation(self):
        val_data = self.load_data(self.val_data_dir, self.preprocessing,'val')
        for case_id, img, mask in val_data:
            self.predict_and_evaluate_case(case_id, img, mask,'val')
        print("----------------------------------------------")
        for tissue_type in self.dice_scores.keys():
            mean_dice = np.mean(self.dice_scores[tissue_type])
            print(f"Mean Dice Score for {tissue_type}: {mean_dice}")
            
            mean_HD = np.mean(self.hausdorff_distances[tissue_type])
            print(f"Mean Hausdorff Distance for {tissue_type}: {mean_HD}")
        print("----------------------------------------------")
        mean_dice_all=np.mean([np.mean(self.dice_scores[tissue]) for tissue in self.dice_scores.keys()])
        mean_HD_all=np.mean([np.mean(self.hausdorff_distances[tissue]) for tissue in self.hausdorff_distances.keys()])
        print(f"Mean DSC: {mean_dice_all}, Hausdorff Distance: {mean_HD_all}")


if __name__ == "__main__":
    preprocessing = Preprocessing(data_aug=False, norm_intensity=True)
    models_path = ["/notebooks/results/DenseUnet_Coronal/1/Best_Model.h5","/notebooks/results/MultiUnet_Coronal/1/Best_Model.h5","/notebooks/results/Unet3D/Best_Model (1).h5"]
    test_data_dir = '/notebooks/data/Training_Set'
    plane_types =["coronal" if 'Coronal' in path else "axial" if "Axial" in path else "3D" for path in models_path]
    prediction_system = Prediction(models_path, test_data_dir, preprocessing, plane_types)
    prediction_system.run_evaluation()
    #prediction_system.predict_test("/notebooks/results/test_predictions")

