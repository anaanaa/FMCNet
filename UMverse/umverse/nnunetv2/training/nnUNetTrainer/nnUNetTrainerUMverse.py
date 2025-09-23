from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import torch
from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans

class nnUNetTrainerUMverse(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 3:
            model = get_umamba_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        
        print("UMambaEnc: {}".format(model))

        return model
