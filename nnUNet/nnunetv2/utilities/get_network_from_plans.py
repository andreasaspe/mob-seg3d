import pydoc
import warnings
from typing import Union
import pickle
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join
import sys

def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None, dataset_name_or_id: str = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    info_dict = {
        'type': type(network),
        'arch_class_name': arch_class_name,
        'input_channels': input_channels,
        'num_classes': output_channels,
        **architecture_kwargs
    }

    # path = "/scratch/awias/data/nnUNet/info_dict_verse.pkl"
    # path = "/scratch/awias/data/Pancreas/info_dict.pkl"
    # path = "/home/awias/data/nnUNet/info_dict_verse.pkl"
    # path = "/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl"
    path = f"/scratch/awias/data/nnUNet/info_dict_{dataset_name_or_id}.pkl"
    try:
        with open(path, 'wb') as handle:
            pickle.dump(info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print(f"ERROR: Peter and I hardcoded this. We try to save the info_dict as a pickle. Please update the path to the correct dataset. Current path defined is: {path}")
        print(f"You can correct it in this file: {__file__}")
        sys.exit(1)
    
    

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


if __name__ == "__main__":
    import torch


    model = get_network_from_plans(
        arch_class_name="dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        arch_kwargs={
            "n_stages": 7,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=4,
        allow_init=True,
        deep_supervision=True,
    )
    data = torch.rand((8, 1, 256, 256))
    target = torch.rand(size=(8, 1, 256, 256))
    outputs = model(data) # this should be a list of torch.Tensor