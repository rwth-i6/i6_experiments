"""
wav2vec-U GAN on the *encoder states of an unsupervised model* instead of the dumped wav2vec
features. Everything else (model, optimizer, schedule, text side, eval) is
:mod:`config_librispeech_960_v1`.

The source model's encoder is run once over the corpus and its per-frame states are written to HDF
(``sis_recipe/dump_features.py``), so GAN training reads plain features and never re-runs the
encoder. The encoder has no subsampling frontend, so the states are frame-synchronous with the
collapsed cluster ids -- same frame rate, and by default the same 512 dims, as the wav2vec features
the baseline uses.

Selecting a different source: add it to :data:`SOURCE_MODELS` (the training name must be registered
in that config's ``model_specs`` and ``checkpoints``) and list it in :data:`VARIANTS`.
"""

import copy
from typing import Dict, List, Optional, Tuple

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from i6_core.returnn.config import ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import (
    build_training_datasets_w_encoder_features,
    build_test_datasets_w_encoder_features,
    encoder_feature_size,
)
from ....unsup_denoising_audio_cluster_and_phoneme.librispeech.configs.config_librispeech_960_w_sil_in_input_v1 import (
    model_specs as unsup_model_specs,
)
from ... import __setup_base_name__
from .....models.recognition.wav2vec_u.decoder_config import DecoderConfig

# reuse the baseline's GAN setup verbatim -- only the speech features differ
from .config_librispeech_960_v1 import (
    base_config,
    settings,
    base_num_epochs,
    get_keep_epochs,
    wav2vec_u_param_groups,
    wav2vec_u_optimizer_class,
)

_UNSUP_PREFIX = "unsup_denoising_audio_cluster_and_phoneme/librispeech/config_librispeech_960_w_sil_in_input_v1"

#: short name -> training_name registered in the source config's ``model_specs`` / ``checkpoints``.
SOURCE_MODELS: Dict[str, str] = {
    # the checkpoint config_librispeech_960_wo_sil_from_unsupervised_ctc_only_v1 also starts from
    "gan-disc-lstm": f"{_UNSUP_PREFIX}/baseline_gan-adv-0.1_disc-lstm_mask-p-0.1-span-1-1_max-num-sil-7_max-surround-1",
    "denoise": f"{_UNSUP_PREFIX}/baseline_max-num-sil-7_max-surround-1",
}

#: (source model, checkpoint epoch, encoder layer; layer None = the last one)
VARIANTS: List[Tuple[str, int, Optional[int]]] = [
    ("gan-disc-lstm", 500, None),
]


def _variant_name(source_name: str, epoch: int, layer: Optional[int]) -> str:
    return f"{source_name}_ep{epoch}" + (f"_layer{layer}" if layer is not None else "")


def py(checkpoints: Dict):
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    for source_name, epoch, layer in VARIANTS:
        training_name = SOURCE_MODELS[source_name]
        assert training_name in unsup_model_specs, (
            f"{training_name!r} not registered -- main() must call the source config's py() first"
        )
        model_spec = unsup_model_specs[training_name]
        checkpoint = checkpoints[training_name][epoch]

        variant = _variant_name(source_name, epoch, layer)
        # the dump jobs are shared between the train and test builders (identical args -> one job)
        dump_alias = f"{prefix_name}/encoder_features/{variant}"

        train_data = build_training_datasets_w_encoder_features(
            settings=settings,
            model_spec=model_spec,
            checkpoint=checkpoint,
            alias_name=dump_alias,
            layer=layer,
        )
        test_data_dict = build_test_datasets_w_encoder_features(
            model_spec=model_spec,
            checkpoint=checkpoint,
            alias_name=dump_alias,
            layer=layer,
        )

        config = dict_update_deep(
            copy.deepcopy(base_config),
            {"model_args.input_dim": encoder_feature_size(model_spec)},
        )

        run_experiment(
            training_name=f"{prefix_name}/baseline_{variant}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            decoder_config=DecoderConfig(),
            additional_configs=[
                ReturnnConfig(
                    config={},
                    python_prolog=[Collection([wav2vec_u_optimizer_class, wav2vec_u_param_groups])],
                )
            ],
        )
