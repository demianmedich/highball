# coding=utf-8
import unittest
from functools import partial

import torch

from highball.config import TrainingConfig
from highball.data.toy_data import (
    ReverseToyDataset,
    ReverseToyDataLoaderConfig
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
            [[61, 26, 41, 26, 46],
             [44, 58, 62, 0, 0],
             [34, 6, 0, 0, 0],
             [37, 44, 20, 16, 0]]
        )
        expected = torch.tensor(
            [[[[0, 1, 1, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1],
               [0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0]]],
             [[[0, 1, 1, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1],
               [0, 0, 0, 1, 1],
               [0, 0, 0, 1, 1]]],
             [[[0, 1, 1, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 1, 1, 1]]],
             [[[0, 1, 1, 1, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 0, 1, 1],
               [0, 0, 0, 0, 1],
               [0, 0, 0, 0, 1]]]]
        )
        pad_token_id = 0
        tgt_mask = TransformerTranslationModel.make_tgt_mask(tgt_seq, pad_token_id)
        print(tgt_mask)

        self.assertEqual(torch.equal(expected, tgt_mask), True)

    def test_transformer_translation_model_toy_dataset_training_step(self):
        vocab = ReverseToyDataset.make_default_vocab()

        batch_size = 4
        num_workers = 2

        dataset_config = ReverseToyDataLoaderConfig(
            batch_size,
            num_workers,
            False,
            False,
            vocab=vocab,
            data_cnt=28,
            max_seq_len=32,
        )

        cfg = TransformerTranslationModelConfig(
            training_cfg=TrainingConfig(
            ),
            optimizer_cfg=None,
            lr_scheduler_cfg=None,
            train_dataloader_cfg=None,
            val_dataloader_cfg=None,
            test_dataloader_cfg=None,
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
            init_method='xavier_normal'
        )
        model = cfg.instantiate()
        # _make_dataloader() is private method.
        dataloader = dataset_config.instantiate()
        dataloader.collate_fn = partial(model.collate_fn,
                                        src_pad_token_id=model.src_pad_token_id,
                                        tgt_pad_token_id=model.tgt_pad_token_id
                                        )
        err_cnt = 0
        try:
            iterator = iter(dataloader)
            items = next(iterator)
            print(items)

            model.training_step(items, 0)
        except Exception as e:
            cause = e.args[0]
            print(cause)
            err_cnt += 1
        print(f'test done. err_cnt {err_cnt}')
        self.assertEqual(err_cnt, 0)


if __name__ == '__main__':
    unittest.main()
