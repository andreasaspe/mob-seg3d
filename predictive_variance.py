import argparse
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import json
import os
import SimpleITK as sitk
import numpy as np
from skimage.measure import label
from pathlib import Path
import os
import torch.nn as nn 
import sys

sys.path.append("/home/awias/code/3DSegRef/uncertainty")
from trainer import Trainer
from model import PartionedCovHead

class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).mu
    
    
def get_model():
    
    # # Best single basis
    # model_kwargs = {
    #     'checkpoint_path': '/scratch/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/checkpoints/exp_basic_run_4_model_epoch_10.pth', #'/scratch/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
    #     'loss_kwargs': {
    #                     'lambda_ce':1.0,
    #                     'lambda_dice':1.0,
    #                     'lambda_nll': 1.0,
    #                     'lambda_kl': 1e-4
    #                 },
    #     'path_to_base': '/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
    #     'num_samples_train': 5,
    #     'num_samples_inference': 30,
    #     'sample_type': 'ours', # Single basis
    #     'dataset_name_or_id': '4'
    # }
    
    
    # # This is PPT
    # model_kwargs = {
    #     'checkpoint_path': '/scratch/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/checkpoints/exp_ppt_run_1_model_epoch_5.pth',
    #     'loss_kwargs': {
    #                     'lambda_ce':1.0,
    #                     'lambda_dice':1.0,
    #                     'lambda_nll': 1.0,
    #                     'lambda_kl': 1e-4
    #                 },
    #     'path_to_base': '/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
    #     'num_samples_train': 5,
    #     'num_samples_inference': 30,
    #     'sample_type': 'torch',  # PPT
    #     'dataset_name_or_id': '4'
    # }


    # This is multi
    model_kwargs = {
        'checkpoint_path': '/scratch/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/checkpoints/exp_multi_basis_proper_gumbel_run_1_model_epoch_18.pth',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 5*1e-4
                    },
        'path_to_base': '/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
        'model_type': 'weighted_basis',
        'cov_weighting_kwargs': {
            'num_bases': 3,
            'sample_type': 'partitioned',
            'class': PartionedCovHead
        },
        'num_samples_train': 5,
        'num_samples_inference': 30,
        'dataset_name_or_id': '4',
    }


    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': '/scratch/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/checkpoints',
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },

        'eval_loader_data_path': '/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs', #Random strange path
        }
    
    trainer = Trainer(model_kwargs, training_kwargs)
    model = WrappedModel(trainer.model)
    return model


def predict_with_nn_unet_on_filelist():
    print("Predict with NN U-Net")
    model_folder = "/scratch/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres"

    input_data_folder = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs"
    output_folder_root = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions"
    
    # Change folder name
    foldername = "man_preds_multibasis_variance" #CHANGE HERE
    output_folder = os.path.join(output_folder_root, foldername)
    
    
    
    
     
    os.makedirs(output_folder, exist_ok=True)

    in_files = []
    out_files = []
    
    for file in os.listdir(input_data_folder):
        if file.endswith("nii.gz"):
            subject = file.split(".nii.gz")[0]
            in_files.append([os.path.join(input_data_folder, file)])
            out_files.append(os.path.join(output_folder, subject + "_pred_var.nii.gz"))
        
    # These have to be set always in my new nnunet format
    os.environ['DO_NOT_USE_SOFTMAX'] = '0'
    os.environ['PREDICT_PIXEL_VARIANCE'] = '0'
    
    print(f"Initializing class")
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=False, # Set to false when predicting variances. No smoothing thank you.
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    print(f"Initializing from trained model folder")

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=[0],
        checkpoint_name='checkpoint_best.pth',
    )



    # Outcommen everything when you have a probabilistic model
    model = get_model()
    predictor.network = model
    predictor.network.get_variance = False
    predictor.network.to(predictor.device)
    predictor.network.eval()
    os.environ['DO_NOT_USE_SOFTMAX'] = '1' # Set to 1 if you DO NOT want to use softmax. This is needed, because our own model DOES take a softmax in the sampling. Basic nnN U-Net does not.
    print(f"Predicting from files")
    predictor.list_of_parameters = [model.state_dict()]
    # CHANGE THIS IF YOU WANT TO PREDICT VARIANCES
    os.environ['PREDICT_PIXEL_VARIANCE'] = '1' # ALWAYS SET TO 1
    
    # Just for the sake of preventing confusion. No need to set DO_NOT_USE_SOFTMAX to 1 and save_probabilities to True when predicting variances. You can, but it is not necessary anymore. Will also not create predictions.
    
    predictor.predict_from_files(in_files,
                                     out_files,
                                     save_probabilities=True, overwrite=True, # Set save_probabilities to True if you want to save the softmax maps
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)







if __name__ == '__main__':
    args = argparse.ArgumentParser(description='dtu-predict-with-nn-unet')
    # config = DTUConfig(args)
    # if config.settings is not None:
    #     # predict_with_nn_unet(config)

    predict_with_nn_unet_on_filelist()
