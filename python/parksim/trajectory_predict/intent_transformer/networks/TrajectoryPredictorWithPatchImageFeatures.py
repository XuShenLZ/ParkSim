from parksim.trajectory_predict.intent_transformer.model_utils import patchify
from parksim.trajectory_predict.intent_transformer.networks.common_blocks import IntentFF, BaseTransformerLightningModule, PositionalEncoding
from parksim.trajectory_predict.intent_transformer.networks.TrajectoryPredictorWithEncoderImageCrossAttention import TransformerWithEncoderImageCrossAttention
from torch import nn
import torch
import torch.nn.functional as F

CNN_OUTPUT_FEATURE_SIZE = 16
TRAJECTORY_FEATURE_SIZE = 3
INTENT_FEATURE_SIZE = 2

DEFAULT_CONFIG = {
    'dropout' : 0.1,
    'num_heads' : 8,
    'num_encoder_layers' : 6,
    'num_decoder_layers' : 6,
    'dim_model' : 64,
    'd_hidden' : 256,
    'patch_size' : 20,
}

class TrajectoryPredictorWithPatchImageFeatures(BaseTransformerLightningModule):
    """
    Creates patches from image features, and linearly encodes them.
    Uses this encoding in the TransformerWithEncoderImageCrossAttention module
    for cross attention.
    """
    def __init__(self, config: dict=DEFAULT_CONFIG, input_shape=(3, 100, 100)):
        super().__init__(config, input_shape)
        self.lr = 6.918309709189363e-05
        self.input_shape=input_shape
        self.dropout=config['dropout']
        self.num_heads=config['num_heads']
        self.num_encoder_layers=config['num_encoder_layers']
        self.num_decoder_layers=config['num_decoder_layers']
        self.dim_model=config['dim_model']
        self.d_hidden=config['d_hidden']
        self.patch_size=config['patch_size']

        assert input_shape[1] == input_shape[2], "Image must be square"
        assert input_shape[1] % self.patch_size == 0, "Patch size must divide the dimensions of the image."

        self.num_patches = (input_shape[1] // self.patch_size) ** 2

        self.positional_encoder = PositionalEncoding(
            d_model=self.dim_model, dropout=0, max_len=5000
        )

        self.intentff = IntentFF(
            d_in=INTENT_FEATURE_SIZE, d_out=self.dim_model, d_hidden=self.d_hidden, dropout_p=self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(self.dim_model, self.num_heads, dropout=self.dropout, activation=F.gelu, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.transformer = TransformerWithEncoderImageCrossAttention(
                                dim_model=self.dim_model, 
                                dim_feature_in=TRAJECTORY_FEATURE_SIZE, 
                                dim_feature_out=TRAJECTORY_FEATURE_SIZE, 
                                num_heads=self.num_heads,
                                num_encoder_layers=self.num_encoder_layers,
                                num_decoder_layers=self.num_decoder_layers,
                                dropout_p=self.dropout)

        self.proj_enc_in = nn.Linear(TRAJECTORY_FEATURE_SIZE, self.dim_model)
        self.proj_dec_in = nn.Linear(TRAJECTORY_FEATURE_SIZE, self.dim_model)
        self.proj_dec_out = nn.Linear(self.dim_model, TRAJECTORY_FEATURE_SIZE)
        self.patch_projection = nn.Linear(self.patch_size * self.patch_size * input_shape[0], self.dim_model)
        self.img_position_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, self.dim_model) * 0.02
        )

    def forward(self, image, trajectories_past, intent, trajectories_future=None, tgt_mask=None):
        """
        image:                  (N, 3, 100, 100)
                                N = batch size
                                Image corresponding to the instance centric view
                                for the current timestep. Agent should be at the
                                center of the image.
                                
        trajectories_past:      (N, T_1, 3)     
                                N = batch size
                                T_1 = timesteps of history
                                3 = (x_coord, y_coord, heading)

        intent:                 (N, 1, 2)
                                N = batch size
                                2 = (x_coord, y_coord)

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
            print(
                "Test time evaluation not yet implemented. Please pass in trajectories_future to train model.")
            return None

        N, T_1, _ = trajectories_past.shape
        _, T_2, _ = trajectories_future.shape
        patches = self.patch_projection(patchify(image, self.patch_size)) + self.img_position_encoding

        img_features = self.encoder(patches)

        intent = self.intentff(intent)

        # trajectory_history: (N, 10, 3)

        # transformer_input: (N, 10, 18)

        src = self.positional_encoder(self.proj_enc_in(trajectories_past))
        tgt = self.positional_encoder(self.proj_dec_in(trajectories_future))

        output = self.transformer(
            src=src, intent=intent, img_features=img_features, tgt=tgt, tgt_mask=tgt_mask)  # (N, T_2, 3)
        return self.proj_dec_out(output)