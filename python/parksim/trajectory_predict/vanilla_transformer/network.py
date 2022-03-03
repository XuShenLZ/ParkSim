from lib2to3.pgen2 import token
from re import M
import torch
from torch import nn
import numpy as np
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        
class SmallRegularizedCNN(nn.Module):
    """
    Simple CNN.
    """
    def __init__(self, output_size=16, dropout_p = 0.2):
        """
        Instantiate the model
        """
        super(SmallRegularizedCNN, self).__init__()
        
        self.image_layers = []

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2),
        ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2),
        ))
        
        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=5),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=3),
            nn.MaxPool2d(2),
        ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=3),
            nn.MaxPool2d(2),
        ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=3),
            nn.MaxPool2d(2),
        ))
        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=1),
            nn.MaxPool2d(2),
        ))

        self.image_layer = nn.Sequential(*self.image_layers)

        self.flatten_layer = nn.Sequential(
            nn.Flatten(),
        )
        
        IMG_LAYER_OUTPUT_SIZE = 16
        NON_SPATIAL_FEATURE_SIZE = 0
        
        
        self.linear_layer1 = nn.Sequential(
            nn.Linear(IMG_LAYER_OUTPUT_SIZE + NON_SPATIAL_FEATURE_SIZE, output_size),
            nn.LayerNorm(output_size)
        )

    def forward(self, img_feature):
        """
        forward method
        """
        x = self.image_layer(img_feature)
        x = self.flatten_layer(x)
        #non_spatial_feature = self.flatten_layer(non_spatial_feature)
        #x = torch.cat([x, non_spatial_feature], 1)
        x = self.linear_layer1(x)
        return x

class TrajectoryPredictTransformerV1(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        # INFO
        CNN_OUTPUT_FEATURE_SIZE = 16
        TRAJECTORY_FEATURE_SIZE = 3 
        self.cnn = SmallRegularizedCNN(output_size=CNN_OUTPUT_FEATURE_SIZE, dropout_p=0.2)
        self.transformer = Transformer(dim_model=16, dim_feature_in=CNN_OUTPUT_FEATURE_SIZE + TRAJECTORY_FEATURE_SIZE, dim_feature_out=TRAJECTORY_FEATURE_SIZE, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dropout_p=0.2)

    def forward(self, instance_centric_img, trajectories_past, trajectories_future=None, tgt_mask=None):
        """
        instance_centric_img:   (N, 1, 400, 400)
                                N = batch size
                                Image corresponding to the instance centric view
                                for the current timestep. Agent should be at the
                                center of the image.
        trajectories_past:      (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)

        trajectories_future:    (N, T_2, 3)     
                                N = batch size
                                T_2 = timesteps of future
                                3 = (x_coord, y_coord, heading)
                                If value is None (such as during test time),
                                then the model will enter predict mode, and
                                generate output appropriately.


        Returns - 

        output:                 (N, T_2, 3)
                                N = batch size
                                T_2 = timesteps of output
                                3 = (x_coord, y_coord, heading)

        """

        if trajectories_future is None:
            print("Test time evaluation not yet implemented. Please pass in trajectories_future to train model.")
            return None

        img_feaure = self.cnn(instance_centric_img) # (N, 16)
        _, T_1, _ = trajectories_past.shape
        _, T_2, _ = trajectories_future.shape
        concat_aligned_img_feature = img_feaure[:, None, :].repeat(1, T_1, 1)
        concatenated_features = torch.concat((trajectories_past, concat_aligned_img_feature), dim=2) # (N, T_1, 3 + 16)
        output = self.transformer(src=concatenated_features, tgt=trajectories_future, tgt_mask=tgt_mask) # (N, T_2, 3)
        return output





class Transformer(nn.Module):
    def __init__(
        self,
        dim_model,
        dim_feature_in,
        dim_feature_out,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            d_model=dim_model, dropout=dropout_p, max_len=5000
        )
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.proj_enc_in = nn.Linear(dim_feature_in, dim_model)
        self.proj_dec_in = nn.Linear(dim_feature_out, dim_model)
        self.out =  nn.Linear(dim_model, dim_feature_out)

    def generate_square_subsequent_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        #mask = mask.fill_diagonal_(float('-inf')) # Convert diagonal to -inf
        
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        src = self.proj_enc_in(src)
        tgt = self.proj_dec_in(tgt)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out

