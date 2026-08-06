"""CNN--Transformer hybrid model for classifying image sequences.

The module combines a convolutional feature extractor with sinusoidal positional
encoding and a Transformer encoder. It supports optional custom pretrained CNN
weights and optional causal attention.
"""

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import math
from Models.Sunspot_Detection import Sunspot_CNN
import warnings
import torch.nn.functional as F


class CNN_part(nn.Module):
    """Expose the feature-extraction portion of a sunspot CNN."""

    def __init__(self, base='resnet18', pretrained=False):
        """Initialize the CNN feature extractor.

        Args:
            base: Backbone architecture passed to :class:`Sunspot_CNN`.
            pretrained: Whether to load the project's custom pretrained weights.
                Custom weights are available for the ``resnet18`` and
                ``resnet152`` backbones.
        """
        super(CNN_part, self).__init__()

        model = Sunspot_CNN(base=base,
                            dropout=0)  # dropout makes no difference here, as we do not use the classifier part of the model

        # this model is used in thesis
        if pretrained:
            if base == 'resnet18':
                state_dict = torch.load("Self_Labeled_Sunspot_resnet18/lr_0.001_ep109.tar")
                new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
                model.load_state_dict(new_state_dict)
                print("Loaded pretrained weights for resnet18")
            elif base == 'resnet152':
                state_dict = torch.load("Self_Labeled_Sunspot_resnet152/lr_0.001_ep63.tar")
                new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
                model.load_state_dict(new_state_dict)
                print("Loaded pretrained weights for resnet152")
            else:
                warnings.warn(f"No custom pretrained weights for base {base}.", UserWarning)

        self.model = model.CNN

    def forward(self, x):
        """Extract spatial features from a batch of images."""
        x = self.model(x)
        return x


class FixedPositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional information to sequence embeddings.

    The encoding has the same dimension as the input embeddings and alternates
    sine and cosine functions at different frequencies, following the original
    Transformer formulation.

    Args:
        d_model: Embedding dimension.
        dropout: Dropout probability applied after adding the encoding.
        max_len: Maximum supported sequence length.
        scale_factor: Multiplier applied to the positional encoding.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        """Initialize and store the fixed positional-encoding matrix.

        Args:
            d_model: Embedding dimension.
            dropout: Dropout probability applied to the encoded input.
            max_len: Maximum supported sequence length.
            scale_factor: Multiplier applied to the sinusoidal encoding.
        """
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        # this stores the variable in the state_dict (used for non-trainable variables)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encodings to ``x`` and apply dropout.

        Args:
            x: Tensor shaped ``[sequence_length, batch_size, embedding_dim]``.

        Returns:
            A tensor with the same shape as ``x``.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def _get_activation_fn(activation):
    """Return the functional implementation of a supported activation."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(
        "activation should be relu/gelu, not {}".format(activation))


class Hybrid(nn.Module):
    """Classify image sequences with a CNN feature extractor and Transformer.

    Each image in a sequence is encoded independently by the CNN. The resulting
    feature sequence is normalized, projected to the Transformer dimension,
    augmented with fixed positional encodings, and processed by the encoder.
    The flattened sequence representation is mapped to two output values.
    """

    def __init__(self, base='resnet18', d_model=32, max_len=50, nhead=4, num_layers=3, dim_feedforward=4 * 32,
                 dropout=0.2, norm_layer=None, activation='relu', norm_first=False, use_causal=False):
        """Initialize the hybrid sequence classifier.

        Args:
            base: CNN backbone used by :class:`Sunspot_CNN`.
            d_model: Transformer embedding dimension.
            max_len: Expected and maximum sequence length.
            nhead: Number of Transformer attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: Hidden dimension of each encoder feed-forward block.
            dropout: Dropout probability used throughout the model.
            norm_layer: Use a final encoder layer normalization when set to
                ``"layer_norm"``; otherwise no final normalization is used.
            activation: Encoder and output activation (``"relu"`` or ``"gelu"``).
            norm_first: Whether encoder layers apply normalization before their
                attention and feed-forward blocks.
            use_causal: Whether to prevent attention to later time steps.
        """
        super(Hybrid, self).__init__()
        self.use_causal = use_causal
        # default output dimension of fundation convolutional neural network
        feature_dim = 1000
        self.d_model = d_model

        # Convolutional Neural Network to extract spatial features from all images
        self.convnet = CNN_part(base=base)

        # Linear Layer to Project input to model dimension
        self.project_input = nn.Linear(feature_dim, d_model)

        # Fixed Positional Encoding for the transformer. The positional encoding is fixed and according to the paper "Attention is all you need"
        self.pos_encodings = FixedPositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len,
                                                     scale_factor=1.0)

        # using layer norm in encoder layer
        if norm_layer == 'layer_norm':
            norm_layer = nn.LayerNorm(d_model)
        else:
            norm_layer = None

        # Transformer Encoder
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                     dropout=dropout, activation=activation, batch_first=False,
                                                     norm_first=norm_first)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
                                          norm=norm_layer)

        # Causal(look - ahead) mask buffer: True above diagonal = disallow attention i→j for j> i
        if self.use_causal:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
                persistent=False
            )

        # NOTE: leave this out for now, but it might come in handy later
        # self.look_ahead_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool().to('cuda')

        # Output Layer
        if activation == "relu":
            self.output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * max_len, 2)
            )
        elif activation == "gelu":
            self.output_layer = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * max_len, 2)
            )
        else:
            raise ValueError(
                "activation should be relu/gelu, not {}".format(activation))

        self.layer_norm = nn.LayerNorm(feature_dim)

        print("\n--------------------------------")
        print(f"Base: {base}")
        print(f"Model Dimension: {d_model}")
        print(f"Number of Transformer Layers: {num_layers}")
        print(f"Number of Attention Heads: {nhead}")
        print(f"Feedforward Dimension: {dim_feedforward}")
        print(f"Dropout: {dropout}")
        print(f"Maximum Sequence Length: {max_len}")
        print("--------------------------------\n")

    def forward(self, x, mask=None):
        """Run an image sequence through the hybrid classifier.

        Args:
            x: Input tensor shaped
                ``[batch_size, channels, sequence_length, height, width]``.
            mask: Padding mask shaped ``[batch_size, sequence_length]``, where
                zero indicates padding and one indicates valid input.

        Returns:
            Tensor shaped ``[batch_size, 2]`` containing classifier outputs.
        """
        B, C, T, H, W = x.shape  #  batch_size, channels, sequence-length, height, width

        x = x.permute(2, 0, 1, 3, 4)  # [T, B, C, H, W]

        x = torch.stack([self.convnet(x[i, :, :, :, :]) for i in range(T)])
        if torch.isnan(x).any():
            raise ValueError("Convnet output contains NaNs")  #  shape [T, B, 1000]

        x = self.layer_norm(x)

        x = self.project_input(x) * math.sqrt(self.d_model)  # [T, B, model_dim] project input to model dimension

        x = self.pos_encodings(x)  # [T, B, model_dim] add positional encodings

        padding_mask = None if mask is None else ~mask.bool()

        if self.use_causal:
            attn_mask = self.causal_mask[:T, :T]

            x = self.encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=padding_mask)

        x = x.reshape(B, -1)  # [batch_size, seq_length*d_model]

        x = self.output_layer(x)

        return x
