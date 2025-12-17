# mob-seg3d

In order to use this code, you also need to take all the required installation steps required by the original nnUNet code. More information can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md. Below is an explicit guide which should get you started:

- Install Pytorch. Can be downloaded here: https://pytorch.org/get-started/locally/. The original nnUNet-repo recommends installing the latest version with support for your hardware (cuda, mps, cpu). The code for the NLDL submission was using version 2.8.0 and Cuda 12.8 and Python 3.13.5.
- Navigate to the nnUNet folder by 'cd nnUNet' and install the editable package by 'pip install -e .'
