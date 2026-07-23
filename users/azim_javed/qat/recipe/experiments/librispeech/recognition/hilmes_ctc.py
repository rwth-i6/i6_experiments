from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import torch

from i6_core.rasr import RasrConfig
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel
from i6_experiments.common.setups.serialization import Collection, Import
from sisyphus import tk

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import vocab_to_bpe_size
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
from ....model_pipelines.common.prior import compute_priors as _compute_priors_base
from ....model_pipelines.common.recog import OfflineRecogParameters, RecogResult, StreamingRecogParameters
from ....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.train import TrainedModel
from .common import BaseRecogVariant, run_single_bpe_variant

# --- Hilmes CTC-specific imports ---
from i6_experiments.users.hilmes.experiments.librispeech.ctc_rnnt_standalone_2024.\
    pytorch_networks.ctc.qat_0711.memristor_v10 import Model as HilmesModel
from i6_experiments.users.hilmes.experiments.librispeech.ctc_rnnt_standalone_2024.\
    pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8


# Patch get_run_ctx since it's not used in inference anyway
try:
    from returnn.torch.context import get_run_ctx 
except ModuleNotFoundError:
    import sys as _sys, types as _types

    _dummy_ctx = _types.ModuleType("returnn.torch.context")
    _dummy_ctx.get_run_ctx = lambda: None
    _sys.modules["returnn.torch.context"] = _dummy_ctx


class HilmesCTCPriorModel(HilmesModel):
    def __init__(self, model_config_dict: dict, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("__fwd_compat_random_arg_080", None)
        super().__init__(model_config_dict=model_config_dict, **kwargs)

    def forward(self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor):
        # translate kwargs to positional args
        log_probs_list, seq_lens = super().forward(audio_samples, audio_samples_size)
        return log_probs_list[0], seq_lens


class HilmesCTCRecogModel(HilmesModel):
    """Hopefully same as QATConformerCTCRecogModel"""

    def __init__(
        self,
        model_config_dict: dict,
        prior_scale: float = 0.0,
        prior_file: Optional[tk.Path] = None,
        blank_penalty: float = 0.0,
        **kwargs,
    ):
        kwargs.pop("device", None)
        kwargs.pop("__fwd_compat_random_arg_080", None)
        super().__init__(model_config_dict=model_config_dict, **kwargs)

        if prior_scale != 0:
            assert prior_file is not None
            self.scaled_priors = prior_scale * torch.tensor(np.loadtxt(prior_file), dtype=torch.float32)
        else:
            self.scaled_priors = torch.zeros((model_config_dict["label_target_size"] + 1,), dtype=torch.float32)
        self.blank_penalty = blank_penalty

    def forward(self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor) -> torch.Tensor:
        log_probs_list, _ = super().forward(audio_samples, audio_samples_size)
        log_probs = log_probs_list[0]
        scores = -log_probs
        scores[:, :, -1] += self.blank_penalty  # blank is the last index
        return scores + self.scaled_priors.to(device=log_probs.device)


@dataclass
class CTCRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    bpe_trafo_lm_scale: float = 0.0
    prior_scale: float = 0.0
    blank_penalty: float = 0.0


def run(
    model: TrainedModel[QuantModelTrainConfigV8],
    variants: Optional[List[CTCRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = librispeech_datasets.EVAL_SETS

    results = []

    for variant in variants:
        results.extend(_run_single_variant(model=model, variant=variant, corpora=corpora))
    return results


def default_recog_variants() -> List[CTCRecogVariant]:
    variants = []
    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for prior_scale in [0.2, 0.3, 0.4, 0.5]:
            variants.append(
                CTCRecogVariant(
                    descriptor=f"tree_4gram_lm{lm_scale}_p{prior_scale}",
                    search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
                        collapse_repeated_labels=True,
                        word_lm_params=librispeech_lm.ArpaLmParams(scale=lm_scale),
                        score_thresholds=[18.0],
                        max_beam_sizes=[2048],
                        word_end_score_threshold=None,
                        max_word_end_beam_size=None,
                    ),
                    prior_scale=prior_scale,
                )
            )
    variants.append(memristor_eq_base_tree_recog_variant())
    # variants.append(
    #     CTCRecogVariant(
    #         descriptor="tree_4gram_lm0.9_p0.2",
    #         search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
    #             collapse_repeated_labels=True,
    #             word_lm_params=librispeech_lm.ArpaLmParams(scale=0.9),
    #             score_thresholds=[18.0],
    #             max_beam_sizes=[2048],
    #             word_end_score_threshold=None,
    #             max_word_end_beam_size=None,
    #         ),
    #         prior_scale=0.2,
    #     )
    # )
    # variants.extend(lstm_tree_recog_variants())
    # variants.extend(lstm_lexfree_recog_variants())
    return variants

    # return [
    #     st14_mb1024_west_off_mweb_off_offline_tree_4gram_recog_variant(),
    #     st18_mb1024_west_off_mweb_off_offline_tree_4gram_recog_variant(),
    #     st14_mb512_west_off_mweb_off_offline_tree_4gram_recog_variant(),
    #     st18_mb512_west_off_mweb_off_offline_tree_4gram_recog_variant(),
    #     st14_mb2048_west_off_mweb_off_offline_tree_4gram_recog_variant(),
    #     st18_mb2048_west_off_mweb_off_offline_tree_4gram_recog_variant(),
    # ]
    # return [
    #     default_offline_lexfree_recog_variant(),
    #     default_offline_lexfree_lstm_recog_variant(),
    #     # default_offline_lexfree_trafo_recog_variant(),
    #     default_offline_tree_recog_variant(),
    #     default_offline_tree_4gram_recog_variant(),
    #     default_offline_tree_lstm_recog_variant(),
    #     default_offline_tree_lstm_4gram_recog_variant(),
    #     # default_offline_tree_trafo_recog_variant(),
    #     # default_offline_tree_trafo_recog_variant_gpu(),
    #     # default_streaming_lexfree_recog_variant(),
    #     # default_streaming_tree_4gram_recog_variant(),
    # ]


def lstm_lexfree_recog_variants() -> List[CTCRecogVariant]:
    variants = []
    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for prior_scale in [0.2, 0.3, 0.4, 0.5]:
            variants.append(
                CTCRecogVariant(
                    descriptor="lexfree_bpe-LSTM_lm{}_p{}_st18.0|14.0_mb2048|256".format(lm_scale, prior_scale),
                    search_algorithm_params=LexiconfreeTimesyncRecogParams(
                        collapse_repeated_labels=True,
                        score_thresholds=[18.0, 14.0],
                        max_beam_sizes=[2048, 256],
                    ),
                    prior_scale=lm_scale,
                    bpe_lstm_lm_scale=prior_scale,
                )
            )
    return variants


def lstm_tree_recog_variants() -> List[CTCRecogVariant]:
    variants = []
    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for prior_scale in [0.2, 0.3, 0.4, 0.5]:
            variants.append(
                CTCRecogVariant(
                    descriptor=f"tree_bpe-LSTM_lm{lm_scale}_p{prior_scale}_st12.0|10.0_mb128|64",
                    search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
                        collapse_repeated_labels=True,
                        score_thresholds=[12.0, 10.0],
                        max_beam_sizes=[128, 64],
                        word_end_score_threshold=0.6,
                        max_word_end_beam_size=16,
                    ),
                    prior_scale=prior_scale,
                    bpe_lstm_lm_scale=lm_scale,
                )
            )
    for lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for prior_scale in [0.2, 0.3, 0.4, 0.5]:
            variants.append(
                CTCRecogVariant(
                    descriptor=f"tree_bpe-LSTM_lm{lm_scale}_p{prior_scale}_st18.0|14.0_mb1024|256",
                    search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
                        collapse_repeated_labels=True,
                        score_thresholds=[18.0, 14.0],
                        max_beam_sizes=[1024, 256],
                        word_end_score_threshold=None,
                        max_word_end_beam_size=None,
                    ),
                    prior_scale=prior_scale,
                    bpe_lstm_lm_scale=lm_scale,
                )
            )
    return variants


def memristor_eq_base_tree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_memristor_eq",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[18.0],
            max_beam_sizes=[1024],
            # no word_lm_params → no LM
        ),
        # word_end_score_threshold and max_word_end_beam_size default to None
    )


def default_offline_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
    )


def default_offline_lexfree_lstm_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree_bpe-LSTM",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0, 12.0],
            max_beam_sizes=[2048, 256],
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_lexfree_trafo_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree_bpe-TrafoLM",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[10.0, 8.0],
            max_beam_sizes=[64, 32],
        ),
        prior_scale=0.2,
        bpe_trafo_lm_scale=0.8,
    )


def default_offline_tree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[6.0],
            max_beam_sizes=[8],
            word_end_score_threshold=0.0,
            max_word_end_beam_size=1,
        ),
    )


def default_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[12.0],  # TODO: Try 14, 18 # first try the thresholds
            max_beam_sizes=[256],  # Try 1024, 512, 2048
            word_end_score_threshold=0.5,  # TODO: find out, should it be None?
            max_word_end_beam_size=16,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def st14_mb1024_west_off_mweb_off_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_st14.0_mb1024_west_off_mweb_off",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[14.0],  # TODO: Try ~14~, 18 # first try the thresholds
            max_beam_sizes=[1024],  # Try ~1024~, 512, 2048
            word_end_score_threshold=None,
            max_word_end_beam_size=None,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def st18_mb1024_west_off_mweb_off_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_st18.0_mb1024_west_off_mweb_off",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[18.0],  # TODO: Try ~14~, ~18~ # first try the thresholds
            max_beam_sizes=[1024],  # Try ~1024~, 512, 2048
            word_end_score_threshold=None,
            max_word_end_beam_size=None,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def st14_mb512_west_off_mweb_off_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_st14.0_mb512_west_off_mweb_off",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[14.0],  # TODO: Try ~14~, 18 # first try the thresholds
            max_beam_sizes=[512],  # Try ~1024~, 512, 2048
            word_end_score_threshold=None,  # TODO: find out, should it be None?
            max_word_end_beam_size=None,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def st18_mb512_west_off_mweb_off_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_st18.0_mb512_west_off_mweb_off",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[18.0],  # TODO: Try ~14~, ~18~ # first try the thresholds
            max_beam_sizes=[512],  # Try ~1024~, 512, 2048
            word_end_score_threshold=None,  # TODO: find out, should it be None?
            max_word_end_beam_size=None,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def st14_mb2048_west_off_mweb_off_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_st14.0_mb2048_west_off_mweb_off",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[14.0],  # TODO: Try ~14~, 18 # first try the thresholds
            max_beam_sizes=[2048],  # Try ~1024~, 512, 2048
            word_end_score_threshold=None,  # TODO: find out, should it be None?
            max_word_end_beam_size=None,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def st18_mb2048_west_off_mweb_off_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_st18.0_mb2048_west_off_mweb_off",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[18.0],  # TODO: Try ~14~, ~18~ # first try the thresholds
            max_beam_sizes=[2048],  # Try ~1024~, 512, 2048
            word_end_score_threshold=None,  # TODO: find out, should it be None?
            max_word_end_beam_size=None,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def default_offline_tree_lstm_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[12.0, 10.0],
            max_beam_sizes=[128, 64],
            word_end_score_threshold=0.6,
            max_word_end_beam_size=16,
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_tree_lstm_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[10.0, 10.0],
            max_beam_sizes=[128, 64],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=32,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.2),
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.6,
    )


def default_offline_tree_trafo_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_trafoLM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[16.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8),
        ),
        search_mode_params=OfflineRecogParameters(),
        prior_scale=0.2,
    )


def default_offline_tree_trafo_recog_variant_gpu() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_trafoLM_gpu",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[16.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8, use_gpu=True, use_kv_cache=False),
        ),
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=24),
        prior_scale=0.2,
    )


def default_streaming_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def default_streaming_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="streaming_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0],
            max_beam_sizes=[1024],
            word_end_score_threshold=0.5,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def _get_model_serializers(
    model: TrainedModel[QuantModelTrainConfigV8], variant: CTCRecogVariant
) -> Collection:
    checkpoint = model.get_checkpoint(variant.epoch)

    prior_file = None
    if variant.prior_scale != 0.0:
        prior_serializers = Collection(
            [
                Import(__name__ + ".HilmesCTCPriorModel"),
                PyTorchModel(
                    model_class_name="HilmesCTCPriorModel",
                    model_kwargs={"model_config_dict": asdict(model.model_config)},
                ),
            ]
        )
        prior_file = _compute_priors_base(
            prior_data_config=librispeech_datasets.get_default_prior_data(),
            model_serializers=prior_serializers,
            checkpoint=checkpoint,
        )

    return Collection(
        [
            Import(__name__ + ".HilmesCTCRecogModel"),
            PyTorchModel(
                model_class_name="HilmesCTCRecogModel",
                model_kwargs={
                    "model_config_dict": asdict(model.model_config),
                    "prior_scale": variant.prior_scale,
                    "prior_file": prior_file,
                    "blank_penalty": variant.blank_penalty,
                },
            ),
        ]
    )


def _get_label_scorer_configs(
    model: TrainedModel[QuantModelTrainConfigV8], variant: CTCRecogVariant
) -> List[RasrConfig]:
    bpe_size = vocab_to_bpe_size(model.model_config.label_target_size)

    label_scorer_configs = [get_no_op_label_scorer_config()]
    if variant.bpe_lstm_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_lstm_label_scorer_config(
                bpe_size=bpe_size,
                scale=variant.bpe_lstm_lm_scale,
                use_gpu=variant.search_mode_params.gpu_mem_rqmt > 0,
            )
        )
    if variant.bpe_trafo_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_transformer_label_scorer_config(
                bpe_size=bpe_size,
                use_gpu=variant.search_mode_params.gpu_mem_rqmt > 0,
                scale=variant.bpe_trafo_lm_scale,
            )
        )

    return label_scorer_configs


def _run_single_variant(
    model: TrainedModel[QuantModelTrainConfigV8],
    variant: CTCRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
) -> List[RecogResult]:
    return run_single_bpe_variant(
        model_descriptor=model.descriptor,
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(model=model, variant=variant),
        label_scorer_configs=_get_label_scorer_configs(model=model, variant=variant),
        bpe_size=vocab_to_bpe_size(model.model_config.label_target_size),
        blank_index=model.model_config.label_target_size,  # blank is at label_target_size (last index)
        sentence_end_index=0 if variant.bpe_lstm_lm_scale != 0 or variant.bpe_trafo_lm_scale != 0 else None,
        variant=variant,
        corpora=corpora,
    )
