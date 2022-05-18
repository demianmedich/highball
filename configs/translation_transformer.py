# coding=utf-8
from highball.config import TrainingConfig
from highball.data.toy_data import (
    ReverseToyDataset,
    ReverseToyDatasetConfig
)
from highball.models.translation_transformer import TransformerTranslationModelConfig
from highball.modules.decoders.transformer_decoder import TransformerDecoderConfig
from highball.modules.encoders.transformer_encoder import TransformerEncoderConfig
from highball.optim_config import AdamOptimizerConfig, SgdOptimizerConfig

vocab = ReverseToyDataset.make_default_vocab()
CFG = TransformerTranslationModelConfig(
    training_cfg=TrainingConfig(
        batch_size=8,
    ),
    optimizer_cfg=AdamOptimizerConfig(
        lr=2e-4,
        weight_decay=0.1,
        betas=(0.9, 0.999)
    ),
    # optimizer_cfg=SgdOptimizerConfig(
    #     lr=2e-4,
    # ),
    lr_scheduler_cfg=None,
    # lr_scheduler_cfg=AnnealingLrSchedulerConfig(
    #     max_lr=1e-3,
    #     min_lr=2e-5,
    #     warmup_steps=500,
    #     decay_steps=4000,
    #     decay_style='cosine',
    # ),
    train_data_cfg=ReverseToyDatasetConfig(
        vocab,
        data_cnt=60000,
        max_seq_len=64,
    ),
    val_data_cfg=ReverseToyDatasetConfig(
        vocab,
        data_cnt=6000,
        max_seq_len=64,
    ),
    test_data_cfg=ReverseToyDatasetConfig(
        vocab,
        data_cnt=6000,
        max_seq_len=64,
    ),
    encoder_cfg=TransformerEncoderConfig(
        len(vocab),
        max_seq_len=64,
        hidden_size=256,
        num_layers=4,
        intermediate_size=1024
    ),
    decoder_cfg=TransformerDecoderConfig(
        len(vocab),
        max_seq_len=64,
        hidden_size=256,
        num_layers=4,
        intermediate_size=1024
    ),
    src_vocab=vocab,
    tgt_vocab=vocab,
    pad_token=vocab.DEFAULT_PAD_TOKEN,
    sos_token=vocab.DEFAULT_BOS_TOKEN,
    eos_token=vocab.DEFAULT_EOS_TOKEN,
)
