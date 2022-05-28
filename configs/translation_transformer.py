# coding=utf-8
from highball.config import (
    TrainingConfig,
    CheckpointingConfig
)
from highball.data.toy_data import (
    ReverseToyDataset,
    ReverseToyDataLoaderConfig
)
from highball.models.translation_transformer import TransformerTranslationModelConfig
from highball.modules.decoders.transformer_decoder import TransformerDecoderConfig
from highball.modules.encoders.transformer_encoder import TransformerEncoderConfig
from highball.optim_config import (
    AdamOptimizerConfig,
    CosineWarmupSchedulerConfig
)

batch_size = 16
num_epochs = 40
num_training_data_cnt = 10000
num_eval_data_cnt = 100
devices = 1
training_steps = int((num_training_data_cnt * num_epochs) // (batch_size * devices))
max_seq_len = 128
vocab = ReverseToyDataset.make_default_vocab()

CFG = TransformerTranslationModelConfig(
    training_cfg=TrainingConfig(
        num_epochs=num_epochs,
        checkpointing_cfg=CheckpointingConfig(
            every_n_epochs=1,
            save_top_k=-1,
        )
    ),
    optimizer_cfg=AdamOptimizerConfig(
        lr=0.01,
        # weight_decay=0.1,
        betas=(0.9, 0.98)
    ),
    lr_scheduler_cfg=CosineWarmupSchedulerConfig(
        warmup_steps=int(training_steps * 0.01),
        training_steps=training_steps
    ),
    train_dataloader_cfg=ReverseToyDataLoaderConfig(
        batch_size,
        8,
        True,
        False,
        vocab,
        data_cnt=num_training_data_cnt,
        max_seq_len=max_seq_len,
    ),
    val_dataloader_cfg=ReverseToyDataLoaderConfig(
        batch_size,
        8,
        False,
        False,
        vocab,
        data_cnt=num_eval_data_cnt,
        max_seq_len=max_seq_len,
    ),
    test_dataloader_cfg=ReverseToyDataLoaderConfig(
        batch_size,
        8,
        False,
        False,
        vocab,
        data_cnt=num_eval_data_cnt,
        max_seq_len=max_seq_len,
    ),
    encoder_cfg=TransformerEncoderConfig(
        len(vocab),
        max_seq_len=max_seq_len,
        hidden_size=64,
        num_layers=2,
        intermediate_size=64 * 4
    ),
    decoder_cfg=TransformerDecoderConfig(
        len(vocab),
        max_seq_len=max_seq_len,
        hidden_size=64,
        num_layers=2,
        intermediate_size=64 * 4
    ),
    src_vocab=vocab,
    tgt_vocab=vocab,
    pad_token=vocab.DEFAULT_PAD_TOKEN,
    sos_token=vocab.DEFAULT_BOS_TOKEN,
    eos_token=vocab.DEFAULT_EOS_TOKEN,
    init_method='xavier_normal'
)
