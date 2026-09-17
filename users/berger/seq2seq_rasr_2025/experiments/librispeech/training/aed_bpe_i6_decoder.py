__all__ = ["run", "get_model_config", "get_train_options"]

from typing import Optional

from i6_models.decoder.attention import (
    AdditiveAttentionConfig as I6AdditiveAttentionConfig,
    AttentionLSTMDecoderV1Config as I6AttentionLSTMDecoderV1Config,
)

from ....data.librispeech.bpe import bpe_to_vocab_size
from ....model_pipelines.aed.pytorch_modules import AEDI6DecoderConfig, AEDI6DecoderModel
from ....model_pipelines.aed.train import AEDTrainOptions, get_train_step_import
from ....model_pipelines.common.train import TrainedModel, train
from . import aed_bpe


def run(
    descriptor: str,
    model_config: Optional[AEDI6DecoderConfig] = None,
    train_options: Optional[AEDTrainOptions] = None,
) -> TrainedModel[AEDI6DecoderConfig]:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    return train(
        descriptor=descriptor,
        model_class=AEDI6DecoderModel,
        model_config=model_config,
        options=train_options,
        train_step_import=get_train_step_import(train_options),
    )


def get_model_config(bpe_size: int = 128) -> AEDI6DecoderConfig:
    base_config = aed_bpe.get_model_config(bpe_size=bpe_size)
    vocab_size = bpe_to_vocab_size(bpe_size=bpe_size)
    decoder_config = base_config.decoder_config

    return AEDI6DecoderConfig(
        logmel_cfg=base_config.logmel_cfg,
        specaug_cfg=base_config.specaug_cfg,
        conformer_cfg=base_config.conformer_cfg,
        final_dropout=base_config.final_dropout,
        enc_dim=base_config.enc_dim,
        decoder_config=I6AttentionLSTMDecoderV1Config(
            encoder_dim=decoder_config.encoder_dim,
            vocab_size=vocab_size,
            target_embed_dim=decoder_config.target_embed_dim,
            target_embed_dropout=decoder_config.target_embed_dropout,
            lstm_hidden_size=decoder_config.lstm_hidden_size,
            zoneout_drop_h=decoder_config.zoneout_drop_h,
            zoneout_drop_c=decoder_config.zoneout_drop_c,
            output_proj_dim=decoder_config.output_proj_dim,
            output_dropout=decoder_config.output_dropout,
            attention_cfg=I6AdditiveAttentionConfig(
                attention_dim=decoder_config.attention_cfg.attention_dim,
                att_weights_dropout=decoder_config.attention_cfg.att_weights_dropout,
            ),
            target_padding_idx=0,
        ),
        label_target_size=base_config.label_target_size,
    )


def get_train_options(bpe_size: int = 128) -> AEDTrainOptions:
    return aed_bpe.get_train_options(bpe_size=bpe_size)
