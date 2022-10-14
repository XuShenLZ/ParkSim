from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_intent import TransformerWithIntent
from parksim.trajectory_predict.intent_transformer.models.base_models import (
    TRAJECTORY_FEATURE_SIZE, INTENT_FEATURE_SIZE, 
    IntentFF, TransformerEncoderLayer, TransformerEncoder, FeatureExtractorCNN
)
from torch import nn
import torch.nn.functional as F

class TrajectoryPredictorWithIntentV2(nn.Module):
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
        self.num_cnn_features=config.get('num_cnn_features', 256)



        self.cnn = FeatureExtractorCNN(input_shape=input_shape,
            num_output_features=self.num_cnn_features, dropout_p=self.dropout, num_conv_layers=self.num_conv_layers)

        self.intentff = IntentFF(
            d_in=INTENT_FEATURE_SIZE, d_out=self.dim_model, d_hidden=self.d_hidden, dropout_p=self.dropout)

        self.image_feat_proj = nn.Linear(self.cnn.feature_size, self.dim_model)

        self.transformer = TransformerWithIntentV2(
                                dim_model=self.dim_model, 
                                dim_feature_in=TRAJECTORY_FEATURE_SIZE, 
                                dim_feature_out=TRAJECTORY_FEATURE_SIZE, 
                                num_heads=self.num_heads,
                                num_encoder_layers=self.num_encoder_layers,
                                num_decoder_layers=self.num_decoder_layers,
                                dropout_p=self.dropout)

        self.proj_enc_in = nn.Linear(TRAJECTORY_FEATURE_SIZE, self.dim_model, bias=False)

    def get_config(self):
        config={
            'dim_model' : self.dim_model,
            'num_heads' : self.num_heads,
            'dropout' : self.dropout,
            'num_encoder_layers' : self.num_encoder_layers,
            'num_decoder_layers' : self.num_decoder_layers,
            'd_hidden' : self.d_hidden,
            'num_cnn_features' : self.num_cnn_features,
            'num_conv_layers' : self.num_conv_layers,
        }
        return config


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

        img_features = self.image_feat_proj(self.cnn(image))

        intent = self.intentff(intent)

        # trajectory_history: (N, 10, 3)

        # transformer_input: (N, 10, 18)

        output = self.transformer(
            src=self.proj_enc_in(trajectories_past), intent=intent, img_features=img_features, tgt=self.proj_enc_in(trajectories_future), tgt_mask=tgt_mask)  # (N, T_2, 3)
        return F.linear(output, self.proj_enc_in.weight.T)

class TransformerWithIntentV2(TransformerWithIntent):
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
        super().__init__(
            dim_model,
            dim_feature_in,
            dim_feature_out,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p,
        )
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
                            d_model=dim_model, 
                            nhead=num_heads, 
                            dropout=dropout_p,
                            batch_first=True)
        encoder_norm = nn.LayerNorm(dim_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)


    def forward(self, src, intent, img_features, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        memory = self.encoder(src, img_features, 
                              src_key_padding_mask=src_pad_mask)
        transformer_out = self.decoder(tgt=tgt, memory=memory, intent=intent, 
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_pad_mask)

        return transformer_out