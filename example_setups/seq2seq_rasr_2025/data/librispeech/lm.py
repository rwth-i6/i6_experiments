__all__ = [
    "ArpaLmParams",
    "TransformerLmParams",
    "WordLmParams",
    "get_word_lm_config",
    "get_bpe_lstm_label_scorer_config",
    "get_bpe_transformer_label_scorer_config",
]

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, Union

from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_experiments.common.datasets.librispeech.vocab import get_lm_vocab
from sisyphus import tk

from ...experiments.librispeech.training import lstm_lm_bpe, transformer_lm_bpe, transformer_lm_word
from ...model_pipelines.lstm_lm.label_scorer_config import get_lstm_lm_label_scorer_config
from ...model_pipelines.lstm_lm.pytorch_modules import LstmLmConfig
from ...model_pipelines.transformer_lm.export import export_model_kv_cached, export_model_stateless
from ...model_pipelines.transformer_lm.label_scorer_config import (
    get_bpe_transformer_lm_label_scorer_config,
)
from ...model_pipelines.transformer_lm.lm_config import get_lm_config_kv_cached, get_lm_config_stateless
from ...model_pipelines.transformer_lm.pytorch_modules import TransformerLmConfig
from ...tools import rasr_binary_path

# =========================
# === Param Dataclasses ===
# =========================


@dataclass
class ArpaLmParams:
    scale: float = 1.0


@dataclass
class TransformerLmParams:
    scale: float = 1.0
    layers: int = 96
    use_kv_cache: bool = True  # Generally, kv cache is slower on GPU because setting up the cache inputs takes so long
    use_gpu: bool = False

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


WordLmParams = Union[ArpaLmParams, TransformerLmParams]

# ===========================
# === LM config factories ===
# ===========================


def get_word_lm_config(lexicon_file: tk.Path, params: WordLmParams) -> RasrConfig:
    if isinstance(params, ArpaLmParams):
        return _get_arpa_lm_config(lexicon_file=lexicon_file, params=params)
    if isinstance(params, TransformerLmParams):
        return _get_transformer_lm_config(params=params)


def get_bpe_lstm_label_scorer_config(bpe_size: int = 128, use_gpu: bool = False, scale: float = 1.0) -> RasrConfig:
    base_config = _get_bpe_lstm_base_label_scorer_config(bpe_size=bpe_size, use_gpu=use_gpu)
    return _copy_with_scale(rasr_config=base_config, scale=scale)


def get_bpe_transformer_label_scorer_config(
    bpe_size: int = 128, num_layers: int = 96, use_gpu: bool = False, scale: float = 1.0
) -> RasrConfig:
    base_config = _get_bpe_transformer_base_label_scorer_config(
        bpe_size=bpe_size,
        num_layers=num_layers,
        use_gpu=use_gpu,
    )
    return _copy_with_scale(rasr_config=base_config, scale=scale)


# =============================
# === Private cache helpers ===
# =============================


def _copy_with_scale(rasr_config: RasrConfig, scale: float) -> RasrConfig:
    rasr_config = copy.deepcopy(rasr_config)
    rasr_config.scale = scale
    return rasr_config


@lru_cache
def _get_arpa_lm_base_config(lexicon_file: tk.Path) -> RasrConfig:
    arpa_lm = get_arpa_lm_dict()["4gram"]
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = arpa_lm
    rasr_config.scale = 1.0

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")  # type: ignore
    crp.language_model_config = rasr_config  # type: ignore
    crp.lexicon_config = RasrConfig()  # type: ignore
    crp.lexicon_config.file = lexicon_file  # type: ignore
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config


def _get_arpa_lm_config(lexicon_file: tk.Path, params: ArpaLmParams) -> RasrConfig:
    base_config = _get_arpa_lm_base_config(lexicon_file=lexicon_file)
    return _copy_with_scale(rasr_config=base_config, scale=params.scale)


@lru_cache
def _get_transformer_lm_artifacts(layers: int) -> Tuple[TransformerLmConfig, PtCheckpoint]:
    train_options = transformer_lm_word.get_train_options()
    model_config = transformer_lm_word.get_model_config()
    model_config.num_layers = layers
    model = transformer_lm_word.run(
        descriptor=f"trafo-lm_word_l-{layers}", train_options=train_options, model_config=model_config
    )
    return model.model_config, model.get_checkpoint()


@lru_cache
def _get_transformer_lm_onnx_model(layers: int, use_kv_cache: bool) -> tk.Path:
    model_config, checkpoint = _get_transformer_lm_artifacts(layers=layers)
    if use_kv_cache:
        return export_model_kv_cached(model_config=model_config, checkpoint=checkpoint)
    return export_model_stateless(model_config=model_config, checkpoint=checkpoint)


@lru_cache
def _get_transformer_lm_base_config(layers: int, use_kv_cache: bool, use_gpu: bool) -> RasrConfig:
    onnx_model = _get_transformer_lm_onnx_model(layers=layers, use_kv_cache=use_kv_cache)
    vocab_file = get_lm_vocab(output_prefix="").vocab

    if use_kv_cache:
        return get_lm_config_kv_cached(onnx_model=onnx_model, vocab_file=vocab_file, use_gpu=use_gpu)
    return get_lm_config_stateless(onnx_model=onnx_model, vocab_file=vocab_file, use_gpu=use_gpu)


def _get_transformer_lm_config(params: TransformerLmParams) -> RasrConfig:
    base_config = _get_transformer_lm_base_config(
        layers=params.layers,
        use_kv_cache=params.use_kv_cache,
        use_gpu=params.use_gpu,
    )
    return _copy_with_scale(rasr_config=base_config, scale=params.scale)


@lru_cache
def _get_bpe_lstm_lm_artifacts(bpe_size: int) -> Tuple[LstmLmConfig, PtCheckpoint]:
    model_config = lstm_lm_bpe.get_model_config(bpe_size=bpe_size)
    train_options = lstm_lm_bpe.get_train_options(bpe_size=bpe_size)

    model = lstm_lm_bpe.run(
        descriptor=f"lstm-lm_bpe-{bpe_size}", model_config=model_config, train_options=train_options
    )
    return model.model_config, model.get_checkpoint()


@lru_cache
def _get_bpe_lstm_base_label_scorer_config(
    bpe_size: int,
    use_gpu: bool,
) -> RasrConfig:
    model_config, checkpoint = _get_bpe_lstm_lm_artifacts(bpe_size=bpe_size)

    return get_lstm_lm_label_scorer_config(
        model_config=model_config,
        checkpoint=checkpoint,
        execution_provider_type="cuda" if use_gpu else None,
    )


@lru_cache
def _get_bpe_transformer_lm_artifacts(bpe_size: int, num_layers: int) -> Tuple[TransformerLmConfig, PtCheckpoint]:
    model_config = transformer_lm_bpe.get_model_config(bpe_size=bpe_size)
    model_config.num_layers = num_layers

    train_options = transformer_lm_bpe.get_train_options(bpe_size=bpe_size)

    model = transformer_lm_bpe.run(
        descriptor=f"trafo-lm_bpe-{bpe_size}_l-{num_layers}", model_config=model_config, train_options=train_options
    )

    return model.model_config, model.get_checkpoint()


@lru_cache
def _get_bpe_transformer_base_label_scorer_config(
    bpe_size: int,
    num_layers: int,
    use_gpu: bool,
) -> RasrConfig:
    model_config, checkpoint = _get_bpe_transformer_lm_artifacts(bpe_size=bpe_size, num_layers=num_layers)

    return get_bpe_transformer_lm_label_scorer_config(
        model_config=model_config,
        checkpoint=checkpoint,
        execution_provider_type="cuda" if use_gpu else None,
    )
