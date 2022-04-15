from lib2to3.pgen2 import token
from re import M
import torch
from torch import nn
import numpy as np
import math

from torch.autograd import Variable

# INFO
CNN_OUTPUT_FEATURE_SIZE = 16
TRAJECTORY_FEATURE_SIZE = 3

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
    def __init__(self, input_shape, output_size=16, dropout_p = 0.2, num_conv_layers=2):
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

        for _ in range(num_conv_layers):
            self.image_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm2d(num_features=8),
                #nn.MaxPool2d(2),
            ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=1),
            nn.MaxPool2d(2),
        ))

        self.image_layer = nn.Sequential(*self.image_layers)

        self.flatten_layer = nn.Sequential(
            nn.Flatten(),
        )
        
        IMG_LAYER_OUTPUT_SIZE = self._get_conv_output_size(input_shape)
        NON_SPATIAL_FEATURE_SIZE = 0
        
        
        self.linear_layer1 = nn.Sequential(
            nn.Linear(IMG_LAYER_OUTPUT_SIZE + NON_SPATIAL_FEATURE_SIZE, output_size),
            nn.LayerNorm(output_size)
        )

    # generate input sample and forward to get shape
    def _get_conv_output_size(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_conv(self, img_feature):
        x = self.image_layer(img_feature)
        x = self.flatten_layer(x)
        return x

    def forward(self, img_feature):
        """
        forward method
        """
        x = self._forward_conv(img_feature)
        #non_spatial_feature = self.flatten_layer(non_spatial_feature)
        #x = torch.cat([x, non_spatial_feature], 1)
        x = self.linear_layer1(x)
        return x

class FeatureExtractorCNN(nn.Module):
    """
    Simple CNN.
    """
    def __init__(self, input_shape, num_output_features=128, dropout_p = 0.2, num_conv_layers=2):
        """
        Instantiate the model
        """
        super(FeatureExtractorCNN, self).__init__()
        
        self.image_layers = []

        self.num_output_features = num_output_features

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, padding='same'),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2),
        ))

        for _ in range(num_conv_layers):
            self.image_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding='same'),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.BatchNorm2d(num_features=8),
                #nn.MaxPool2d(2),
            ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=num_output_features, kernel_size=3, padding='same'),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=num_output_features),
            nn.MaxPool2d(2),
        ))

        self.image_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=num_output_features, out_channels=num_output_features, kernel_size=3, padding='same'),
            nn.Dropout(dropout_p),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm2d(num_features=num_output_features),
        ))

        #Height and width divided by 4 because of 2 max pools
        channels, height, width = input_shape
        self.feature_size = (height // 4) * (width // 4)
        
        self.image_layer = nn.Sequential(*self.image_layers)

    # generate input sample and forward to get shape
    def _get_conv_output_size(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_conv(self, img_feature):
        x = self.image_layer(img_feature)
        return x

    def forward(self, img_feature):
        """
        forward method
        """
        N, _, _, _ = img_feature.shape
        x = self._forward_conv(img_feature)
        x = x.reshape(N, self.num_output_features, self.feature_size)
        return x

class TrajectoryPredictTransformerV1(nn.Module):
    def __init__(
        self, input_shape, dropout=0.2, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, dim_model=16, num_conv_layers=2,
    ):
        super().__init__()

        self.cnn = SmallRegularizedCNN(input_shape=input_shape, output_size=CNN_OUTPUT_FEATURE_SIZE, dropout_p=dropout, num_conv_layers=num_conv_layers)
        self.transformer = Transformer(dim_model=dim_model, 
                                        dim_feature_in=CNN_OUTPUT_FEATURE_SIZE + TRAJECTORY_FEATURE_SIZE, 
                                        dim_feature_out=TRAJECTORY_FEATURE_SIZE, 
                                        num_heads=num_heads, 
                                        num_encoder_layers=num_encoder_layers, 
                                        num_decoder_layers=num_decoder_layers, 
                                        dropout_p=dropout)

    def forward(self, images_past, trajectories_past, trajectories_future=None, tgt_mask=None):
        """
        images_past:            (N, T_1, 3, 100, 100)
                                N = batch size
                                Image history corresponding to the instance centric view
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

        N, T_1, _ = trajectories_past.shape
        _, T_2, _ = trajectories_future.shape

        concat_aligned_img_feature = torch.empty(
            size=(N, T_1, CNN_OUTPUT_FEATURE_SIZE)).to(trajectories_past.device)

        # img (N, T_1, 3, 100, 100) -> CNN -> (N, T_1, 16)
        for t in range(T_1):
            concat_aligned_img_feature[:, t, :] = self.cnn(images_past[:, t, :, :, :])


        # trajectory_history: (N, 10, 3)

        # transformer_input: (N, 10, 19)

        concatenated_features = torch.cat((trajectories_past, concat_aligned_img_feature), dim=2) # (N, T_1, 3 + 16)
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

