# coding=utf-8
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    Optional,
    Any,
    Union,
)

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from highball.config import (
    LightningModuleConfig,
)
from highball.modules.decoders.transformer_decoder import TransformerDecoderConfig
from highball.modules.encoders.transformer_encoder import TransformerEncoderConfig
from highball.utils import (
    INIT_METHOD,
    init_tensor_
)
from highball.vocab import (
    SimpleVocabulary,
    load_vocabulary,
)


@dataclass
class TransformerTranslationModelConfig(LightningModuleConfig):
    encoder_cfg: TransformerEncoderConfig
    decoder_cfg: TransformerDecoderConfig
    src_vocab: Union[str, SimpleVocabulary]
    tgt_vocab: Union[str, SimpleVocabulary]
    pad_token: str
    sos_token: str
    eos_token: str
    init_method: INIT_METHOD

    def instantiate(self) -> "TransformerTranslationModel":
        if type(self.src_vocab) == str:
            self.src_vocab = load_vocabulary(self.src_vocab)
        if type(self.tgt_vocab) == str:
            self.tgt_vocab = load_vocabulary(self.tgt_vocab)
        return TransformerTranslationModel(self)


class TransformerTranslationModel(LightningModule):

    def __init__(self, cfg: TransformerTranslationModelConfig) -> None:
        super().__init__()
        self.cfg = deepcopy(cfg)

        self.src_vocab = self.cfg.src_vocab
        self.tgt_vocab = self.cfg.tgt_vocab
        self.src_pad_token_id = self.src_vocab.idx(self.cfg.pad_token)
        self.tgt_pad_token_id = self.tgt_vocab.idx(self.cfg.pad_token)
        self.sos_token_id = self.tgt_vocab.idx(self.cfg.sos_token)
        self.eos_token_id = self.tgt_vocab.idx(self.cfg.eos_token)

        self.encoder = self.cfg.encoder_cfg.instantiate()
        self.decoder = self.cfg.decoder_cfg.instantiate()
        self.generator = Generator(self.cfg.decoder_cfg.hidden_size,
                                   self.cfg.decoder_cfg.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        init_method = self.cfg.init_method
        if isinstance(module, nn.Linear):
            init_tensor_(init_method, module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            init_tensor_(init_method, module.weight)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        if self.cfg.optimizer_cfg is None:
            return None
        optimizer = self.cfg.optimizer_cfg.instantiate(self.parameters())

        if self.cfg.lr_scheduler_cfg is None:
            return optimizer
        else:
            scheduler = self.cfg.lr_scheduler_cfg.instantiate(optimizer)
            return [optimizer], [scheduler]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.cfg.train_dataloader_cfg is None:
            return []
        dataloader: DataLoader = self.cfg.train_dataloader_cfg.instantiate()
        dataloader.collate_fn = partial(self.collate_fn,
                                        src_pad_token_id=self.src_pad_token_id,
                                        tgt_pad_token_id=self.tgt_pad_token_id)
        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.cfg.val_dataloader_cfg is None:
            return []
        dataloader: DataLoader = self.cfg.val_dataloader_cfg.instantiate()
        dataloader.collate_fn = partial(self.collate_fn,
                                        src_pad_token_id=self.src_pad_token_id,
                                        tgt_pad_token_id=self.tgt_pad_token_id)
        return dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.cfg.test_dataloader_cfg is None:
            return []
        dataloader: DataLoader = self.cfg.test_dataloader_cfg.instantiate()
        dataloader.collate_fn = partial(self.collate_fn,
                                        src_pad_token_id=self.src_pad_token_id,
                                        tgt_pad_token_id=self.tgt_pad_token_id)
        return dataloader

    def encode(self, src_seq: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        return self.encoder(src_seq, src_mask)

    def decode(
            self,
            tgt_seq: Tensor,
            enc_output: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        decoder_output = self.decoder(tgt_seq, enc_output, src_mask, tgt_mask)
        return decoder_output

    def forward(self, input_dict: dict) -> Any:
        """forward method is for inference"""
        src_seq: Tensor = input_dict['src_seq']
        device = src_seq.device
        src_mask: Tensor = input_dict.get('src_mask', None)
        if src_mask is not None and src_mask.dim() == 2:
            src_mask = src_mask[:, None, None, :]

        enc_output = self.encode(src_seq, src_mask)

        batch_size = src_seq.shape[0]
        pred = torch.ones(batch_size, 1, dtype=torch.long, device=device) * self.sos_token_id
        decoded_logits = torch.zeros(batch_size, 1, self.cfg.decoder_cfg.vocab_size, device=device)

        for i in range(self.cfg.decoder_cfg.max_seq_len):
            tgt_mask = self.make_tgt_mask(pred, self.tgt_pad_token_id)
            decoded_output = self.decode(pred, enc_output, src_mask, tgt_mask)
            decoded_last_logit = self.generator(decoded_output[:, -1]).unsqueeze(1)
            decoded_ids = torch.argmax(decoded_last_logit, dim=-1)
            pred = torch.cat([pred, decoded_ids], dim=-1)
            decoded_logits = torch.cat([decoded_logits, decoded_last_logit], dim=1)

        output = {
            'enc_output': enc_output,
            'pred': pred[:, 1:],
            'logits': decoded_logits[:, 1:]
        }
        return output

    def training_step(self, batched_input: dict, batch_idx: int) -> STEP_OUTPUT:
        src_seq: Tensor = batched_input['src_seq']
        src_mask: Tensor = batched_input.get('src_mask', None)
        if src_mask is not None and src_mask.dim() == 2:
            src_mask = src_mask[:, None, None, :]
        tgt_seq: Tensor = batched_input['tgt_seq']

        # encode source sequence
        encoder_output = self.encode(src_seq, src_mask)

        # make decoder input sequence
        # 1. change EOS token to PAD token (every seq has EOS token)
        eos_pos = (tgt_seq == self.eos_token_id).type_as(tgt_seq)
        decoder_input_seq = tgt_seq * (1 - eos_pos) + (eos_pos * self.tgt_pad_token_id)
        # 2. insert SOS token at head of decoder input sequence
        decoder_input_seq = F.pad(decoder_input_seq[:, :-1], [1, 0], value=self.sos_token_id)
        # 3. make tgt_mask
        tgt_mask = self.make_tgt_mask(decoder_input_seq, self.tgt_pad_token_id)

        decoder_output = self.decode(decoder_input_seq, encoder_output, src_mask, tgt_mask)
        decoder_logits = self.generator(decoder_output)
        loss = self.calculate_loss(decoder_logits, tgt_seq, self.tgt_pad_token_id)
        acc = self.calculate_accuracy(torch.argmax(decoder_logits, dim=-1), tgt_seq,
                                      self.tgt_pad_token_id)
        self.log('training_loss', loss)
        self.log('training_acc', acc, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batched_input: dict, batch_idx: int) -> Optional[STEP_OUTPUT]:
        tgt_seq = batched_input['tgt_seq']
        tgt_seq_len = tgt_seq.shape[1]
        output = self(batched_input)

        pred = output['pred']
        logits = output['logits']

        loss = self.calculate_loss(logits[:, :tgt_seq_len], tgt_seq, self.tgt_pad_token_id)
        acc = self.calculate_accuracy(pred[:, :tgt_seq_len], tgt_seq, self.tgt_pad_token_id)
        # on_epoch: Automatically accumulates and logs at the end of the epoch
        # reduce_fx: Reduction function over step values for end of epoch. Uses torch.mean() by default.
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        })
        return None

    def test_step(self, batched_input: dict, batch_idx: int) -> Optional[STEP_OUTPUT]:
        tgt_seq = batched_input['tgt_seq']
        tgt_seq_len = tgt_seq.shape[1]
        output = self(batched_input)

        pred = output['pred']
        logits = output['logits']

        loss = self.calculate_loss(logits[:, :tgt_seq_len], tgt_seq, self.tgt_pad_token_id)
        acc = self.calculate_accuracy(pred[:, :tgt_seq_len], tgt_seq, self.tgt_pad_token_id)
        # on_epoch: Automatically accumulates and logs at the end of the epoch
        # reduce_fx: Reduction function over step values for end of epoch. Uses torch.mean() by default.
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc
        })
        return None

    @staticmethod
    def calculate_loss(logits: Tensor, target: Tensor, pad_token_id: int) -> Tensor:
        last_dim = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, last_dim), target.reshape(-1),
                               ignore_index=pad_token_id)
        return loss

    @staticmethod
    def calculate_accuracy(pred: Tensor, target: Tensor, pad_token_id: int) -> Tensor:
        accuracy = pred == target
        mask = target != pad_token_id
        accuracy &= mask
        return accuracy.sum() / mask.sum()

    @staticmethod
    def collate_fn(samples: list, src_pad_token_id: int, tgt_pad_token_id: int):
        src_seq = [torch.tensor(e['src_seq'], dtype=torch.long) for e in samples]
        tgt_seq = [torch.tensor(e['tgt_seq'], dtype=torch.long) for e in samples]
        src_seq = torch.nn.utils.rnn.pad_sequence(src_seq, batch_first=True,
                                                  padding_value=src_pad_token_id)
        src_mask = (src_seq == src_pad_token_id).to(dtype=torch.float32)
        tgt_seq = torch.nn.utils.rnn.pad_sequence(tgt_seq, batch_first=True,
                                                  padding_value=tgt_pad_token_id)
        return {
            'src_seq': src_seq.contiguous(),
            'src_mask': src_mask.contiguous(),
            'tgt_seq': tgt_seq.contiguous(),
        }

    @staticmethod
    def make_tgt_mask(tgt_seq: Tensor, pad_token_id: int) -> Tensor:
        """make look ahead + padding mask

        Args:
            tgt_seq: target sequence tensor with shape [batch_size, seq_len]
            pad_token_id: padding token idx

        Returns:
            Target mask with shape [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len = tgt_seq.shape
        dtype = tgt_seq.dtype
        device = tgt_seq.device
        pad_mask = (tgt_seq == pad_token_id).unsqueeze(1)
        look_ahead_mask = torch.triu(
            torch.ones(batch_size, seq_len, seq_len, dtype=dtype, device=device),
            diagonal=1
        )
        # tgt_mask = torch.clip(pad_mask + look_ahead_mask, min=0., max=1.)
        tgt_mask = pad_mask | look_ahead_mask
        return tgt_mask.unsqueeze(1)


class Generator(nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, decoder_output: Tensor) -> Tensor:
        return self.linear(decoder_output)
