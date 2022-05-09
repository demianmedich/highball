# coding=utf-8
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor

from highball.modules.activation import ACTIVATION_LAYERS
from highball.modules.vanilla_transformer import (
    TransformerDecoderCore, PositionalEncoding, PositionEmbedding
)


@dataclass
class TransformerDecoderConfig:
    vocab_size: int
    max_seq_len: int
    use_positional_encoding: bool = True
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 2048
    layernorm_eps: float = 0.1
    attention_dropout_prob: float = 0.1
    mlp_dropout_prob: float = 0.1
    mlp_activation: ACTIVATION_LAYERS = 'relu'

    def instantiate(self) -> "TransformerDecoder":
        return TransformerDecoder(self)


class TransformerDecoder(nn.Module):
    """Transformer decoder"""

    def __init__(self, cfg: TransformerDecoderConfig) -> None:
        super().__init__()
        self.cfg = deepcopy(cfg)

        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        if self.cfg.use_positional_encoding:
            self.pos_embedding = PositionalEncoding(self.cfg.hidden_size,
                                                    max_len=self.cfg.max_seq_len)
        else:
            self.pos_embedding = PositionEmbedding(self.cfg.hidden_size, self.cfg.max_seq_len)

        self.decoder_core = TransformerDecoderCore(
            self.cfg.hidden_size,
            self.cfg.num_layers,
            self.cfg.num_heads,
            self.cfg.intermediate_size,
            self.cfg.layernorm_eps,
            self.cfg.attention_dropout_prob,
            self.cfg.mlp_dropout_prob,
            self.cfg.mlp_activation
        )

    def forward(
            self,
            tgt_seq: Tensor,
            enc_output: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt_embed = self.token_embedding(tgt_seq)
        tgt_embed = self.pos_embedding(tgt_embed)
        dec_output = self.decoder_core(tgt_embed, enc_output, src_mask, tgt_mask)
        return dec_output
