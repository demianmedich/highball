# coding=utf-8
import unittest

import torch

from highball.config import TrainingConfig
from highball.data.toy_data import (
    ReverseToyDataset,
    ReverseToyDatasetConfig
)
from highball.models.translation_transformer import (
    TransformerTranslationModelConfig,
    TransformerTranslationModel
)
from highball.modules.decoders.transformer_decoder import TransformerDecoderConfig
from highball.modules.encoders.transformer_encoder import TransformerEncoderConfig


class TranslationModelTestCase(unittest.TestCase):

    def test_make_target_mask(self):
        tgt_seq = torch.tensor(
            [[61, 26, 41, 26, 46, 33, 36, 60, 11, 8, 19, 56, 17, 24, 12, 13, 29, 2],
             [44, 58, 62, 46, 45, 4, 49, 14, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [34, 6, 7, 54, 43, 17, 52, 5, 36, 27, 2, 0, 0, 0, 0, 0, 0, 0],
             [37, 44, 20, 16, 44, 42, 32, 6, 59, 53, 58, 30, 20, 9, 2, 0, 0, 0]])
        pad_token_id = 0
        tgt_mask = TransformerTranslationModel.make_tgt_mask(tgt_seq, pad_token_id)
        print(tgt_mask)

    def test_transformer_translation_model_toy_dataset_training_step(self):
        vocab = ReverseToyDataset.make_default_vocab()

        dataset_config = ReverseToyDatasetConfig(
            vocab=vocab,
            data_cnt=28,
            max_seq_len=32,
        )

        cfg = TransformerTranslationModelConfig(
            training_cfg=TrainingConfig(
                batch_size=4
            ),
            optimizer_cfg=None,
            lr_scheduler_cfg=None,
            train_data_cfg=None,
            val_data_cfg=None,
            test_data_cfg=None,
            encoder_cfg=TransformerEncoderConfig(
                len(vocab),
                max_seq_len=32,
                use_positional_encoding=False
            ),
            decoder_cfg=TransformerDecoderConfig(
                len(vocab),
                max_seq_len=32,
                use_positional_encoding=False
            ),
            src_vocab=vocab,
            tgt_vocab=vocab,
            pad_token=vocab.DEFAULT_PAD_TOKEN,
            sos_token=vocab.DEFAULT_BOS_TOKEN,
            eos_token=vocab.DEFAULT_EOS_TOKEN,
        )
        model = cfg.instantiate()
        # _make_dataloader() is private method.
        dataloader = model._make_dataloader(dataset_config, shuffle=False)
        iterator = iter(dataloader)
        items = next(iterator)
        print(items)

        model.training_step(items, 0)


if __name__ == '__main__':
    unittest.main()
