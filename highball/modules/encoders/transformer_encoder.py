# coding=utf-8
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from torch import (
    nn,
    Tensor
)

from highball.modules.activation import ACTIVATION_LAYERS
from highball.modules.vanilla_transformer import (
    TransformerEncoderCore,
    PositionalEncoding,
    PositionEmbedding
)


@dataclass
class TransformerEncoderConfig:
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

    def instantiate(self) -> "TransformerEncoder":
        return TransformerEncoder(self)


class TransformerEncoder(nn.Module):
    """Transformer encoder"""

    def __init__(self, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = deepcopy(cfg)

        self.token_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        if self.cfg.use_positional_encoding:
            self.pos_embedding = PositionalEncoding(self.cfg.hidden_size,
                                                    max_len=self.cfg.max_seq_len)
        else:
            self.pos_embedding = PositionEmbedding(self.cfg.hidden_size, self.cfg.max_seq_len)

        self.encoder_core = TransformerEncoderCore(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            intermediate_size=cfg.intermediate_size,
            layernorm_eps=cfg.layernorm_eps,
            attention_dropout_prob=cfg.attention_dropout_prob,
            mlp_dropout_prob=cfg.mlp_dropout_prob,
            mlp_activation=cfg.mlp_activation
        )

    def forward(self, src_seq: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src_embed = self.token_embedding(src_seq)
        src_embed = self.pos_embedding(src_embed)
        output = self.encoder_core(src_embed, src_mask)
        return output
