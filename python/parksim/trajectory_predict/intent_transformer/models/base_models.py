import torch
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable
from parksim.trajectory_predict.vanilla_transformer.network import SmallRegularizedCNN, FeatureExtractorCNN
from parksim.trajectory_predict.intent_transformer.model_utils import patchify

import math
import copy

import torch.nn.functional as F
from torch.autograd import Variable


# INFO
CNN_OUTPUT_FEATURE_SIZE = 16
TRAJECTORY_FEATURE_SIZE = 3
TRANSFORMER_FEATURE_SIZE = 16
INTENT_FEATURE_SIZE = 2
DROPOUT_P = 0.2


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SmallRegularizedCNN(nn.Module):
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

class PatchCNN(nn.Module):
    """
    Simple CNN.
    """
    def __init__(self, input_shape, num_output_features=128, dropout_p = 0.2, num_conv_layers=2):
        """
        Instantiate the model
        """
        super(PatchCNN, self).__init__()
        
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

