# coding=utf-8
import math
from typing import (
    Optional,
    Tuple,
    Callable
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from highball.modules.activation import (
    activation_layer,
    ACTIVATION_LAYERS,
)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model: int, dropout_prob: float = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)


class PositionEmbedding(nn.Module):
    """Position embedding"""

    def __init__(self, hidden_size: int, max_seq_len: int, dropout_prob: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        pos = torch.arange(start=0, end=self.max_seq_len).unsqueeze(0)
        self.register_buffer('pos', pos)

    def forward(self, x: Tensor):
        x = x + self.pos_embedding(self.pos)
        return self.dropout(x)


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size: int, layernorm_eps: float = 1e-6, dropout_prob: float = 0.1):
        """Sub-layer connection to pass through layernorm and residual-connection"""
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor, sublayer: Callable) -> Tensor:
        return x + self.dropout(sublayer(self.layernorm(x)))


class MultiHeadedAttention(nn.Module):
    """Multi-headed Attention"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, 'hidden_size should be divisible by num_heads'
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.concat_fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.depth: int = hidden_size // num_heads
        self.num_heads = num_heads
        self._attention_weight: Optional[Tensor] = None

    @property
    def attention_weight(self):
        return self._attention_weight

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = split_heads(q, self.num_heads, self.depth)
        k = split_heads(k, self.num_heads, self.depth)
        v = split_heads(v, self.num_heads, self.depth)

        attention_output, self._attention_weight = scaled_dot_product_attention(
            q, k, v, attention_mask=attention_mask, dropout=self.dropout
        )
        attention_output = combine_heads(attention_output)
        attention_output = self.concat_fc(attention_output)
        return attention_output


def split_heads(x: Tensor, num_heads: int, depth: int) -> Tensor:
    """Split input tensor into different heads.

    Args:
        x: Input tensor with shape [batch_size, seq_len, hidden_size]
        num_heads: The number of heads
        depth: hidden_size // num_heads

    Returns:
        A tensor with shape [batch_size, num_heads, seq_len, depth]
    """
    batch_size, seq_len, hidden_dim = x.shape
    x = x.view(batch_size, seq_len, num_heads, depth).permute(0, 2, 1, 3).contiguous()
    return x


def combine_heads(x: Tensor) -> Tensor:
    """Combine input tensor that has been split.

    Args:
        x: Input tensor with shape [batch_size, num_heads, seq_len, depth]
    """
    batch_size, num_heads, seq_len, depth = x.shape
    x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * depth)
    return x


def scaled_dot_product_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
) -> Tuple[Tensor, Tensor]:
    """scaled dot product attention

    Args:
        query: Query tensor with shape [batch_size, num_heads, seq_len, hidden_dim // num_heads]
        key: Key tensor with shape [batch_size, num_heads, seq_len, hidden_dim // num_heads]
        value: Same as key argument.
        attention_mask (opt): attention mask with shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        dropout (opt): Dropout for attention weight

    Returns:
        Attention output with shape [batch_size, num_heads, seq_len, depth] and weight
    """
    dk = query.shape[-1]
    qk_transposed = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dk)

    if attention_mask is not None:
        assert attention_mask.dim() == 4, f'attention_mask should be 4 dimensional tensor but got {attention_mask.dim()}'
        qk_transposed += (attention_mask * -1e9)

    attention_weight = F.softmax(qk_transposed, dim=-1)
    if dropout is not None:
        attention_weight = dropout(attention_weight)

    attention_output = torch.matmul(attention_weight, value)
    return attention_output, attention_weight


class Mlp(nn.Module):
    """Multi-layer perceptron for Transformer sub-layer"""

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            activation: ACTIVATION_LAYERS = 'relu',
            dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = activation_layer(activation)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Transformer encoder layer"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            layernorm_eps: float = 0.1,
            attention_dropout_prob: float = 0.1,
            mlp_dropout_prob: float = 0.1,
            mlp_activation: ACTIVATION_LAYERS = 'relu'
    ):
        super().__init__()
        self.self_attention_layer = MultiHeadedAttention(hidden_size, num_heads,
                                                         dropout_prob=attention_dropout_prob)
        self.self_attention_conn = SublayerConnection(hidden_size, layernorm_eps=layernorm_eps,
                                                      dropout_prob=attention_dropout_prob)
        self.mlp_layer = Mlp(hidden_size, intermediate_size, activation=mlp_activation)
        self.mlp_conn = SublayerConnection(hidden_size,
                                           layernorm_eps=layernorm_eps,
                                           dropout_prob=mlp_dropout_prob)

    def forward(self, src_seq: Tensor, attention_mask: Optional[Tensor] = None):
        output = self.self_attention_conn(
            src_seq,
            lambda x: self.self_attention_layer(x, x, x, attention_mask=attention_mask)
        )
        output = self.mlp_conn(output, self.mlp_layer)
        return output


class TransformerEncoderCore(nn.Module):
    """Transformer encoder"""

    def __init__(
            self,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            intermediate_size: int,
            layernorm_eps: float = 0.1,
            attention_dropout_prob: float = 0.1,
            mlp_dropout_prob: float = 0.1,
            mlp_activation: ACTIVATION_LAYERS = 'relu'
    ) -> None:
        super().__init__()
        layers = [EncoderLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            layernorm_eps=layernorm_eps,
            attention_dropout_prob=attention_dropout_prob,
            mlp_dropout_prob=mlp_dropout_prob,
            mlp_activation=mlp_activation
        ) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.output_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_eps)

    def forward(self, src_embed: Tensor, attention_mask: Optional[Tensor] = None):
        x = src_embed
        for layer in self.layers:
            x = layer(x, attention_mask)
        output = self.output_layernorm(x)
        return output


class DecoderLayer(nn.Module):
    """Transformer decoder layer"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            layernorm_eps: float = 0.1,
            attention_dropout_prob: float = 0.1,
            mlp_dropout_prob: float = 0.1,
            mlp_activation: ACTIVATION_LAYERS = 'relu'
    ):
        super().__init__()
        self.self_attention_layer = MultiHeadedAttention(hidden_size, num_heads,
                                                         dropout_prob=attention_dropout_prob)
        self.self_attention_conn = SublayerConnection(hidden_size,
                                                      layernorm_eps=layernorm_eps,
                                                      dropout_prob=attention_dropout_prob)
        self.enc_dec_attention_layer = MultiHeadedAttention(hidden_size, num_heads,
                                                            dropout_prob=attention_dropout_prob)
        self.enc_dec_attention_conn = SublayerConnection(hidden_size,
                                                         layernorm_eps=layernorm_eps,
                                                         dropout_prob=attention_dropout_prob)
        self.mlp_layer = Mlp(hidden_size, intermediate_size, activation=mlp_activation)
        self.mlp_conn = SublayerConnection(hidden_size,
                                           layernorm_eps=layernorm_eps,
                                           dropout_prob=mlp_dropout_prob)

    def forward(
            self,
            tgt_seq: Tensor,
            encoder_output: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.self_attention_conn(
            tgt_seq,
            lambda x: self.self_attention_layer(x, x, x, attention_mask=tgt_mask)
        )
        output = self.enc_dec_attention_conn(
            output,
            lambda x: self.enc_dec_attention_layer(x, encoder_output, encoder_output,
                                                   attention_mask=src_mask)
        )
        output = self.mlp_conn(output, self.mlp_layer)
        return output


class TransformerDecoderCore(nn.Module):
    """Transformer decoder"""

    def __init__(
            self,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            intermediate_size: int,
            layernorm_eps: float = 0.1,
            attention_dropout_prob: float = 0.1,
            mlp_dropout_prob: float = 0.1,
            mlp_activation: ACTIVATION_LAYERS = 'relu'
    ) -> None:
        super().__init__()
        layers = [DecoderLayer(
            hidden_size,
            num_heads,
            intermediate_size,
            layernorm_eps=layernorm_eps,
            attention_dropout_prob=attention_dropout_prob,
            mlp_dropout_prob=mlp_dropout_prob,
            mlp_activation=mlp_activation
        ) for _ in range(num_layers)]

        self.layers = nn.ModuleList(layers)
        self.output_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_eps)

    def forward(
            self,
            tgt_embed: Tensor,
            enc_output: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt_embed
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        output = self.output_layernorm(x)
        return output
