import torch
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable
from parksim.trajectory_predict.vanilla_transformer.network import SmallRegularizedCNN, FeatureExtractorCNN
from parksim.trajectory_predict.intent_transformer.model_utils import patchify

import math
import copy

import torch.nn.functional as F


# INFO
CNN_OUTPUT_FEATURE_SIZE = 16
TRAJECTORY_FEATURE_SIZE = 3
TRANSFORMER_FEATURE_SIZE = 16
INTENT_FEATURE_SIZE = 2
DROPOUT_P = 0.2


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
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

class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, img_features: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, img_features, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.img_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, img_features: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._image_feature_mha_block(self.norm3(x), img_features)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm3(x + self._image_feature_mha_block(x, img_features))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    # intent attention block
    def _image_feature_mha_block(self, x: Tensor, img_features: Tensor,
                          attn_mask: Optional[Tensor]=None, key_padding_mask: Optional[Tensor]=None) -> Tensor:
        x = self.img_multihead_attn(x, img_features, img_features,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout3(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, 
                d_model: int, 
                nhead: int, 
                dim_feedforward: int = 2048, 
                dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                layer_norm_eps: float = 1e-5, 
                batch_first: bool = False, norm_first: bool = False,
                device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        self.intent_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm4 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, 
                memory: Tensor, 
                intent: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            intent: processed intent vector (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask,
                                   tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory,
                                    memory_mask, memory_key_padding_mask)
            x = x + self._intent_mha_block(self.norm3(x), intent,
                                    memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm4(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory,
                           memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._intent_mha_block(x, intent,
                           memory_mask, memory_key_padding_mask))
            x = self.norm4(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # intent attention block
    def _intent_mha_block(self, x: Tensor, intent: Tensor,
                          attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.intent_multihead_attn(x, intent, intent,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout3(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)

class TransformerDecoder(nn.Module):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, intent: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            intent: encoding of intent (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, intent, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class IntentFF(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 16, dropout_p: float = 0.1):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(p=dropout_p)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(d_hidden, d_out),
            nn.LayerNorm(d_out),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, intent: Tensor):
        x = self.linear1(intent)
        x = self.linear2(x)

        return x


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


class TransformerWithIntentV3(TransformerWithIntent):
    def __init__(
        self,
        dim_model,
        dim_feature_in,
        dim_feature_out,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        num_patches
    ):
        super().__init__(
            dim_model,
            dim_feature_in,
            dim_feature_out,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_p
        )
        self.num_patches = num_patches
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

class TrajectoryPredictorWithIntentV3(nn.Module):
    def __init__(self, config: dict, input_shape=(3, 100, 100)):
        super().__init__()

        self.input_shape=input_shape
        self.dropout=config.get('dropout', 0.1)
        self.num_heads=config.get('num_heads', 8)
        self.num_encoder_layers=config.get('num_encoder_layers', 6)
        self.num_decoder_layers=config.get('num_decoder_layers', 6)
        self.dim_model=config.get('dim_model', 64)
        self.d_hidden=config.get('d_hidden', 256)
        self.patch_size=config.get('patch_size', 25)

        assert input_shape[1] == input_shape[2], "Image must be square"
        assert input_shape[1] % self.patch_size == 0, "Patch size must divide the dimensions of the image."

        self.num_patches = (input_shape[1] // self.patch_size) ** 2



        #self.cnn = FeatureExtractorCNN(input_shape=input_shape,
        #    num_output_features=self.num_cnn_features, dropout_p=self.dropout, num_conv_layers=self.num_conv_layers)

        self.intentff = IntentFF(
            d_in=INTENT_FEATURE_SIZE, d_out=self.dim_model, d_hidden=self.d_hidden, dropout_p=self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(self.dim_model, self.num_heads, dropout=self.dropout, activation=F.gelu, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.transformer = TransformerWithIntentV3(
                                dim_model=self.dim_model, 
                                dim_feature_in=TRAJECTORY_FEATURE_SIZE, 
                                dim_feature_out=TRAJECTORY_FEATURE_SIZE, 
                                num_heads=self.num_heads,
                                num_encoder_layers=self.num_encoder_layers,
                                num_decoder_layers=self.num_decoder_layers,
                                dropout_p=self.dropout,
                                num_patches=self.num_patches)

        self.proj_enc_in = nn.Linear(TRAJECTORY_FEATURE_SIZE, self.dim_model, bias=False)
        self.patch_projection = nn.Linear(self.patch_size * self.patch_size * input_shape[0], self.dim_model)
        self.img_position_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, self.dim_model) * 0.02
        )

    def get_config(self):
        config={
            'dim_model' : self.dim_model,
            'num_heads' : self.num_heads,
            'dropout' : self.dropout,
            'num_encoder_layers' : self.num_encoder_layers,
            'num_decoder_layers' : self.num_decoder_layers,
            'd_hidden' : self.d_hidden,
            'patch_size' : self.patch_size,
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
        patches = self.patch_projection(patchify(image, self.patch_size)) + self.img_position_encoding

        img_features = self.encoder(patches)

        intent = self.intentff(intent)

        # trajectory_history: (N, 10, 3)

        # transformer_input: (N, 10, 18)

        output = self.transformer(
            src=self.proj_enc_in(trajectories_past), intent=intent, img_features=img_features, tgt=self.proj_enc_in(trajectories_future), tgt_mask=tgt_mask)  # (N, T_2, 3)
        return F.linear(output, self.proj_enc_in.weight.T)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])