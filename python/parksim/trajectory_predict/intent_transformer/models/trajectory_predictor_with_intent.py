from parksim.trajectory_predict.intent_transformer.models.base_models import (
    CNN_OUTPUT_FEATURE_SIZE, TRAJECTORY_FEATURE_SIZE, INTENT_FEATURE_SIZE, 
    SmallRegularizedCNN, IntentFF, PositionalEncoding, TransformerDecoderLayer, 
    TransformerDecoder,
)
import torch
from torch import nn

class TrajectoryPredictorWithIntent(nn.Module):
    def __init__(self, config: dict, input_shape=(3, 100, 100)):
        super().__init__()

        self.input_shape=input_shape
        self.dropout=config.get('dropout', 0.1)
        self.num_heads=config.get('num_heads', 8)
        self.num_encoder_layers=config.get('num_encoder_layers', 6)
        self.num_decoder_layers=config.get('num_decoder_layers', 6)
        self.dim_model=config.get('dim_model', 64)
        self.d_hidden=config.get('d_hidden', 256)
        self.num_conv_layers=config.get('num_conv_layers', 2)

        self.cnn = SmallRegularizedCNN(input_shape=input_shape,
            output_size=CNN_OUTPUT_FEATURE_SIZE, dropout_p=self.dropout, num_conv_layers=self.num_conv_layers)

        self.intentff = IntentFF(
            d_in=INTENT_FEATURE_SIZE, d_out=self.dim_model, d_hidden=self.d_hidden, dropout_p=self.dropout)

        self.transformer = TransformerWithIntent(
                                dim_model=self.dim_model, 
                                dim_feature_in=CNN_OUTPUT_FEATURE_SIZE + TRAJECTORY_FEATURE_SIZE, 
                                dim_feature_out=TRAJECTORY_FEATURE_SIZE, 
                                num_heads=self.num_heads,
                                num_encoder_layers=self.num_encoder_layers,
                                num_decoder_layers=self.num_decoder_layers,
                                dropout_p=self.dropout)

    def get_config(self):
        config={
            'dim_model' : self.dim_model,
            'num_heads' : self.num_heads,
            'dropout' : self.dropout,
            'num_encoder_layers' : self.num_encoder_layers,
            'num_decoder_layers' : self.num_decoder_layers,
            'd_hidden' : self.d_hidden,
            'num_conv_layers' : self.num_conv_layers,
        }
        return config

    def forward(self, images_past, trajectories_past, intent, trajectories_future=None, tgt_mask=None):
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

        intent:                 (N, 2) or (N, 1, 2) ?
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

        concat_aligned_img_feature = torch.empty(
            size=(N, T_1, CNN_OUTPUT_FEATURE_SIZE)).to(trajectories_past.device)

        # img (N, T_1, 3, 100, 100) -> CNN -> (N, T_1, 16)
        for t in range(T_1):
            concat_aligned_img_feature[:, t, :] = self.cnn(
                images_past[:, t, :, :, :])

        intent = self.intentff(intent)

        # trajectory_history: (N, 10, 3)

        # transformer_input: (N, 10, 18)

        concatenated_features = torch.cat(
            (trajectories_past, concat_aligned_img_feature), dim=2)  # (N, T_1, 2 + 16)
        output = self.transformer(
            src=concatenated_features, intent=intent, tgt=trajectories_future, tgt_mask=tgt_mask)  # (N, T_2, 3)
        return output

class TransformerWithIntent(nn.Module):
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

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=dim_model, 
                            nhead=num_heads, 
                            dropout=dropout_p,
                            batch_first=True)
        encoder_norm = nn.LayerNorm(dim_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        # Transfromer Decoder
        decoder_layer = TransformerDecoderLayer(
                            d_model=dim_model, 
                            nhead=num_heads,
                            dropout=dropout_p,
                            batch_first=True)
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)
        
        self.proj_enc_in = nn.Linear(dim_feature_in, dim_model)
        self.proj_dec_in = nn.Linear(dim_feature_out, dim_model)
        self.out = nn.Linear(dim_model, dim_feature_out)

    def generate_square_subsequent_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        # Lower triangular matrix
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float(
            '-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        #mask = mask.fill_diagonal_(float('-inf')) # Convert diagonal to -inf

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def forward(self, src, intent, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        src = self.proj_enc_in(src)
        tgt = self.proj_dec_in(tgt)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        memory = self.encoder(src,
                              src_key_padding_mask=src_pad_mask)
        transformer_out = self.decoder(tgt=tgt, memory=memory, intent=intent, 
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)

        return out