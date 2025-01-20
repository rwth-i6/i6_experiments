from typing import List, Optional

import torch
from i6_experiments.common.datasets.tedlium2.corpus import get_corpus_object_dict
from i6_models.assemblies.conformer.conformer_v2 import ConformerBlockV2Config, ConformerEncoderV2Config
from i6_models.config import ModuleFactoryV1
from i6_models.parts.conformer import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from sisyphus import tk
from ...model_pipelines.common.corpus import ScorableCorpus

from ...data.tedlium2.bpe import (
    bpe_to_vocab_size,
    get_bpe_vocab_file,
)
from ...data.tedlium2.datasets import (
    get_default_bpe_cv_data,
    get_default_bpe_train_data,
    get_default_prior_data,
    get_default_recog_data,
)
from ...data.tedlium2.lexicon import get_bpe_word_lexicon
from ...data.tedlium2.lm import get_binary_lm
from ...model_pipelines.bpe_ffnn_transducer.pipeline import PipelineConfig, run_pipeline
from ...model_pipelines.bpe_ffnn_transducer.pytorch_modules import (
    FFNNTransducerConfig,
    SpecaugmentByLengthConfig,
)
from ...model_pipelines.bpe_ffnn_transducer.subroutines.configs import (
    CTCFlashlightRecogRoutineConfig,
    CTCPriorRoutineConfig,
    CTCRasrRecogRoutineConfig,
    RecogRoutineConfig,
    TrainRoutineConfig,
    TransducerRasrRecogRoutineConfig,
)
from ...model_pipelines.common.learning_rates import OCLRConfig


def get_baseline_config() -> PipelineConfig:
    bpe_size = 128
    vocab_size = bpe_to_vocab_size(bpe_size)

    model_config = FFNNTransducerConfig(
        logmel_cfg=LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
            n_fft=400,
        ),
        specaug_cfg=SpecaugmentByLengthConfig(
            start_epoch=5,
            time_min_num_masks=2,
            time_max_mask_per_n_frames=25,
            time_mask_max_size=20,
            freq_min_num_masks=2,
            freq_max_num_masks=5,
            freq_mask_max_size=16,
        ),
        conformer_cfg=ConformerEncoderV2Config(
            num_layers=12,
            frontend=ModuleFactoryV1(
                GenericFrontendV1,
                GenericFrontendV1Config(
                    in_features=80,
                    layer_ordering=[
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Activation,
                        FrontendLayerType.Pool2d,
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Activation,
                        FrontendLayerType.Pool2d,
                    ],
                    conv_kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],
                    conv_out_dims=[32, 64, 64, 32],
                    conv_strides=None,
                    conv_paddings=None,
                    pool_kernel_sizes=[(2, 1), (2, 1)],
                    pool_strides=None,
                    pool_paddings=None,
                    activations=[torch.nn.ReLU(), torch.nn.ReLU()],
                    out_features=512,
                ),
            ),
            block_cfg=ConformerBlockV2Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=512,
                    hidden_dim=2048,
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=512,
                    num_att_heads=8,
                    att_weights_dropout=0.1,
                    dropout=0.1,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=512,
                    kernel_size=31,
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                    norm=LayerNormNC(512),
                ),
                modules=["ff", "conv", "mhsa", "ff"],
                scales=[0.5, 1.0, 1.0, 0.5],
            ),
        ),
        dropout=0.1,
        enc_dim=512,
        enc_output_indices=[5, 11],
        pred_num_layers=2,
        pred_dim=640,
        pred_activation=torch.nn.Tanh(),
        context_history_size=1,
        context_embedding_dim=256,
        joiner_dim=1024,
        joiner_activation=torch.nn.Tanh(),
        target_size=vocab_size + 1,
    )

    train_routine_config = TrainRoutineConfig(
        train_data_config=get_default_bpe_train_data(bpe_size),
        cv_data_config=get_default_bpe_cv_data(bpe_size),
        save_epochs=list(range(20, 18, 20)) + list(range(180, 201, 4)),
        batch_frames=12_000,
        accum_grad_multiple_step=2,
        lr_config=OCLRConfig(
            init_lr=7e-06,
            peak_lr=5e-04,
            decayed_lr=5e-05,
            final_lr=1e-07,
            inc_epochs=96,
            dec_epochs=96,
            final_epochs=8,
        ),
        enc_loss_scales={5: 0.3, 11: 0.7},
        gradient_clip=1.0,
        weight_decay=0.001,
    )

    ctc_prior_routine_config = CTCPriorRoutineConfig(
        enc_layer=11,
        prior_data_config=get_default_prior_data(),
        batch_frames=20_000,
    )

    def get_transducer_rasr_recog_routine_config(
        corpus_name: str,
        descriptor: str,
        epoch: int = train_routine_config.save_epochs[-1],
        lexicon_file: Optional[tk.Path] = None,
        lm_file: Optional[tk.Path] = None,
        lm_scale: float = 0.0,
        blank_penalty: float = 1.0,
        max_beam_size: int = 1,
        top_k_tokens: Optional[int] = None,
        score_threshold: Optional[float] = None,
        ilm_scale: float = 0.2,
    ) -> TransducerRasrRecogRoutineConfig:
        return TransducerRasrRecogRoutineConfig(
            descriptor=descriptor,
            corpus=ScorableCorpus(
                corpus_name=corpus_name,
                bliss_corpus_file=get_corpus_object_dict(audio_format="wav", output_prefix="corpora")[
                    corpus_name
                ].corpus_file,
            ),
            recog_data_config=get_default_recog_data(corpus_name),
            blank_index=vocab_size,
            epoch=epoch,
            lexicon_file=lexicon_file,
            lm_file=lm_file,
            vocab_file=get_bpe_vocab_file(bpe_size=bpe_size, add_blank=True),
            device="cpu",
            lm_scale=lm_scale,
            ilm_scale=ilm_scale,
            blank_penalty=blank_penalty,
            max_beam_size=max_beam_size,
            top_k_tokens=top_k_tokens,
            score_threshold=score_threshold,
        )

    def get_ctc_rasr_recog_routine_config(
        corpus_name: str,
        descriptor: str,
        epoch: int = train_routine_config.save_epochs[-1],
        enc_layer: int = 11,
        lexicon_file: Optional[tk.Path] = None,
        lm_file: Optional[tk.Path] = None,
        prior_scale: float = 0.0,
        lm_scale: float = 0.0,
        blank_penalty: float = 0.0,
        max_beam_size: int = 1,
        top_k_tokens: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> CTCRasrRecogRoutineConfig:
        return CTCRasrRecogRoutineConfig(
            enc_layer=enc_layer,
            descriptor=descriptor,
            corpus=ScorableCorpus(
                corpus_name=corpus_name,
                bliss_corpus_file=get_corpus_object_dict(audio_format="wav", output_prefix="corpora")[
                    corpus_name
                ].corpus_file,
            ),
            recog_data_config=get_default_recog_data(corpus_name),
            blank_index=vocab_size,
            epoch=epoch,
            prior_config=ctc_prior_routine_config,
            lexicon_file=lexicon_file,
            lm_file=lm_file,
            vocab_file=get_bpe_vocab_file(bpe_size=bpe_size, add_blank=True),
            device="cpu",
            prior_scale=prior_scale,
            lm_scale=lm_scale,
            blank_penalty=blank_penalty,
            max_beam_size=max_beam_size,
            top_k_tokens=top_k_tokens,
            score_threshold=score_threshold,
        )

    def get_ctc_flashlight_recog_routine_config(
        corpus_name: str,
        descriptor: str,
        epoch: int = train_routine_config.save_epochs[-1],
        enc_layer: int = 11,
        lexicon_file: Optional[tk.Path] = None,
        lm_file: Optional[tk.Path] = None,
        prior_scale: float = 0.0,
        lm_scale: float = 0.0,
        blank_penalty: float = 0.0,
        beam_size: int = 1,
        beam_size_token: Optional[int] = None,
        beam_threshold: float = float("inf"),
    ) -> CTCFlashlightRecogRoutineConfig:
        return CTCFlashlightRecogRoutineConfig(
            enc_layer=enc_layer,
            descriptor=descriptor,
            corpus=ScorableCorpus(
                corpus_name=corpus_name,
                bliss_corpus_file=get_corpus_object_dict(audio_format="wav", output_prefix="corpora")[
                    corpus_name
                ].corpus_file,
            ),
            recog_data_config=get_default_recog_data(corpus_name),
            blank_index=vocab_size,
            epoch=epoch,
            prior_config=ctc_prior_routine_config,
            vocab_file=get_bpe_vocab_file(bpe_size=bpe_size, add_blank=True),
            lexicon_file=lexicon_file,
            lm_file=lm_file,
            beam_size=beam_size,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            lm_scale=lm_scale,
            device="cpu",
            prior_scale=prior_scale,
            blank_penalty=blank_penalty,
        )

    recognitions: List[RecogRoutineConfig] = []

    for epoch in train_routine_config.save_epochs:
        recognitions.append(
            get_transducer_rasr_recog_routine_config(
                descriptor=f"intermediate_e-{epoch}",
                corpus_name="dev",
                epoch=epoch,
            )
        )

    for corpus_name in ["dev", "test"]:
        recognitions.append(
            get_transducer_rasr_recog_routine_config(descriptor="rasr_greedy", corpus_name=corpus_name, max_beam_size=1)
        )

        recognitions.append(
            get_transducer_rasr_recog_routine_config(descriptor="rasr_beam-8", corpus_name=corpus_name, max_beam_size=8)
        )

        recognitions.append(
            get_ctc_rasr_recog_routine_config(descriptor="ctc_rasr_greedy", corpus_name=corpus_name, max_beam_size=1)
        )

        recognitions.append(
            get_ctc_flashlight_recog_routine_config(
                descriptor="ctc_flashlight_greedy", corpus_name=corpus_name, beam_size=1
            )
        )

        recognitions.append(
            get_ctc_flashlight_recog_routine_config(
                descriptor="ctc_flashlight_beam-8",
                corpus_name=corpus_name,
                beam_size=8,
                lexicon_file=get_bpe_word_lexicon(bpe_size),
                lm_file=get_binary_lm("4gram"),
                prior_scale=0.3,
                lm_scale=2.0,
            )
        )

    return PipelineConfig(
        model_config=model_config,
        train_config=train_routine_config,
        recog_configs=recognitions,
    )


def run_bpe_ffnn_transducer_baseline() -> None:
    run_pipeline(get_baseline_config(), prefix="tedlium2")
