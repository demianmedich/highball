# coding=utf-8
from highball.config import TrainingConfig
from highball.data.toy_data import (
    ReverseToyDataset,
    ReverseToyDatasetConfig
)
from highball.models.translation_transformer import TransformerTranslationModelConfig
from highball.modules.decoders.transformer_decoder import TransformerDecoderConfig
from highball.modules.encoders.transformer_encoder import TransformerEncoderConfig
from highball.optim_config import (
    AdamOptimizerConfig,
    CosineWarmupSchedulerConfig
)

batch_size = 8
num_epochs = 40
num_training_data_cnt = 60000
training_steps = int(num_training_data_cnt * num_epochs // batch_size)
vocab = ReverseToyDataset.make_default_vocab()

CFG = TransformerTranslationModelConfig(
    training_cfg=TrainingConfig(
        num_epochs=num_epochs,
        batch_size=8,
        num_workers=1,
    ),
    optimizer_cfg=AdamOptimizerConfig(
        lr=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.999)
    ),
    lr_scheduler_cfg=CosineWarmupSchedulerConfig(
        warmup_steps=int(training_steps * 0.01),
        training_steps=training_steps
    ),
    train_data_cfg=ReverseToyDatasetConfig(
        vocab,
        data_cnt=num_training_data_cnt,
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
        hidden_size=512,
        num_layers=6,
        intermediate_size=2048
    ),
    decoder_cfg=TransformerDecoderConfig(
        len(vocab),
        max_seq_len=64,
        hidden_size=512,
        num_layers=6,
        intermediate_size=2048
    ),
    src_vocab=vocab,
    tgt_vocab=vocab,
    pad_token=vocab.DEFAULT_PAD_TOKEN,
    sos_token=vocab.DEFAULT_BOS_TOKEN,
    eos_token=vocab.DEFAULT_EOS_TOKEN,
)
