from parksim.trajectronpp.models.common_blocks import BaseMGCVAELightningModule
import torch
from torch import nn
import torch.nn.functional as F

class MGCVAE(nn.Module):
    # TODO: fill this in
    # TODO: figure out what hyperparams this takes
    def __init__(self) -> None:
        super().__init__()

        # INFO
        # TODO: update these for accuracy
        self.model_type = "Multimodal Generative CVAE"
        self.dim_model = 2

        # LAYERS
        # TODO: update w layers from paper
        self.layer1 = nn.Linear(3, 20)
        self.layer2 = nn.Linear(20, 3)

    def  forward(self, target_history):
        middle = self.layer1(target_history)
        out = self.layer2(middle)
        return out

# TODO: fill this in
DEFAULT_CONFIG = {}

class TrajectoryPredictorMGCVAE(BaseMGCVAELightningModule):
    # TODO: fill this in
    def __init__(self, config: dict, input_shape=(), loss_fn=F.l1_loss) -> None:
        super().__init__(config, input_shape, loss_fn)
        # TODO: include the other models & update this one
        self.model = MGCVAE()
    
    def forward(self, ego_history, target_history, neighbor_veh_history, neighbor_ped_history, semantic_map):
        # TODO: update this to be accurate
        # TODO: do i need to take the target future?
        """
        ego_history:            (N, T_1, 3)
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)
                                
        target_history:         (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)
                                
        neighbor_veh_history:   (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)

        neighbor_ped_history:   (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)

        semantic_map:           (N, 3, 100, 100)     
                                N = batch size
                                Image corresponding to instance centric view at current time step


        Returns - 

        output:                 (N, T_2, 3)
                                N = batch size
                                T_2 = timesteps of output
                                3 = (x_coord, y_coord, heading)

        """
        # TODO: really need to update this one
        output = self.model(target_history)
        return output
