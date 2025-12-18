# mob-seg3d

In order to use this code, you also need to take all the required installation steps required by the original nnUNet code. More information can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md. Below is an explicit guide which should get you started by following the same steps as was done in the paper:

- Install Pytorch. Can be downloaded here: https://pytorch.org/get-started/locally/. The original nnUNet-repo recommends installing the latest version with support for your hardware (cuda, mps, cpu). The code for the NLDL submission was using version 2.8.0 and Cuda 12.8 and Python 3.13.5.
- Navigate to the nnUNet folder by 'cd nnUNet' and install the editable package by 'pip install -e .'

# Step 0: Download the data
The TotalSegmentator dataset can be downloaded here:
https://zenodo.org/records/10047292

# Step 1: Training the deterministic nnUNet

First you have prepare the totalsegmentator dataset and put it into the right format for the nnUNet dataset.

This can be done by the script prepare_totalseg.py:
python prepare_totalseg.py \
  --base_dir /data/Totalsegmentator_dataset_v201 \
  --nnunet_raw /data/nnUNet_raw \
  --organ Pancreas \
  --dataset_id 1

You can then setup training of the nnUNet according to the original authors. Below is a describtion of how it was done for this paper:

Following the nnUNet required data structure you'll have to define the following environment variables:

export nnUNet_raw="PATH-TO-DATASET/nnUNet_raw"
export nnUNet_preprocessed="PATH-TO-DATASET/nnUNet_preprocessed"
export nnUNet_results="PATH-TO-DATASET/nnUNet_results"

Extract dataset fingerprint:
nnUNetv2_extract_fingerprint -d <dataset_id> -verify_dataset_integrity -verbose -pl nnUNetPlannerResEncL -c 3d_fullres

Plan the experiment:
nnUNetv2_plan_experiment -d <dataset_id> -c 3d_fullres -pl nnUNetPlannerResEncL -np 4

Preprocess the Dataset:
nnUNetv2_preprocess -d <dataset_id> -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8

Train the model (choose a fold)
nnUNetv2_train <dataset_id> 3d_fullres <fold> -tr nnUNetTrainerNoMirroring -p nnUNetResEncUNetLPlans

The model will not 


# Step 2: Training the stochastic nnUNet



# Step 3: Get predictive variance