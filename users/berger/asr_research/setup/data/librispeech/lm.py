__all__ = [
    "ArpaLmParams",
    "TransformerLmParams",
    "KazukiTrafoLmParams",
    "OnnxTokenLmParams",
    "WordLmParams",
    "get_word_lm_config",
    "get_word_transformer_onnx_model",
    "get_bpe_lstm_label_scorer_config",
    "get_bpe_transformer_label_scorer_config",
]

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union

from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_core.returnn import Checkpoint, PtCheckpoint
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
    lm_file: Optional[tk.Path] = None


@dataclass
class TransformerLmParams:
    scale: float = 1.0
    layers: int = 96
    use_kv_cache: bool = True  # Generally, kv cache is slower on GPU because setting up the cache inputs takes so long
    use_gpu: bool = False
    onnx_model: Optional[tk.Path] = None
    vocab_file: Optional[tk.Path] = None
    # None retains the existing backend default. For onnx-stateless, 1 also
    # disables speculative queuing (workaround for retained discarded histories).
    max_batch_size: Optional[int] = None

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


@dataclass
class KazukiTrafoLmParams:
    scale: float = 1.0


@dataclass
class OnnxTokenLmParams:
    """External subword LM with explicit output-index names in the lexicon."""

    onnx_model: tk.Path
    vocab_file: tk.Path
    scale: float = 1.0
    use_gpu: bool = True
    max_batch_size: int = 16


WordLmParams = Union[ArpaLmParams, TransformerLmParams, KazukiTrafoLmParams, OnnxTokenLmParams]

# ===========================
# === LM config factories ===
# ===========================


def get_word_lm_config(
    lexicon_file: tk.Path, params: WordLmParams, rasr_binary_path_override: Optional[tk.Path] = None
) -> RasrConfig:
    if isinstance(params, OnnxTokenLmParams):
        config = get_lm_config_kv_cached(params.onnx_model, params.vocab_file, use_gpu=params.use_gpu)
        config.scale = params.scale
        config.max_batch_size = params.max_batch_size
        # No invented UNK: every syntactic token is an explicitly checked output id.
        # RASR's default is empty. A bare `vocab-unknown-word =` is invalid
        # configuration syntax; remove the helper's <UNK> override instead.
        del config.vocab_unknown_word
        return config
    if isinstance(params, ArpaLmParams):
        if rasr_binary_path_override is not None:
            arpa_lm = params.lm_file if params.lm_file is not None else get_arpa_lm_dict()["4gram"]
            return _copy_with_scale(
                _get_arpa_lm_base_config(lexicon_file, arpa_lm, rasr_binary_path_override), params.scale
            )
        return _get_arpa_lm_config(lexicon_file=lexicon_file, params=params)
    if isinstance(params, TransformerLmParams):
        return _get_transformer_lm_config(params=params)
    if isinstance(params, KazukiTrafoLmParams):
        return _get_kazuki_trafo_lm_config(params=params)


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
def _get_arpa_lm_base_config(
    lexicon_file: tk.Path, arpa_lm: tk.Path, rasr_binary_path_override: Optional[tk.Path] = None
) -> RasrConfig:
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = arpa_lm
    rasr_config.scale = 1.0

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = (rasr_binary_path_override or rasr_binary_path).join_right("lm-util.linux-x86_64-standard")  # type: ignore
    crp.language_model_config = rasr_config  # type: ignore
    crp.lexicon_config = RasrConfig()  # type: ignore
    crp.lexicon_config.file = lexicon_file  # type: ignore
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config


def _get_arpa_lm_config(lexicon_file: tk.Path, params: ArpaLmParams) -> RasrConfig:
    arpa_lm = params.lm_file if params.lm_file is not None else get_arpa_lm_dict()["4gram"]
    base_config = _get_arpa_lm_base_config(lexicon_file=lexicon_file, arpa_lm=arpa_lm)
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
    # checkpoint = PtCheckpoint(
    #     tk.Path(
    #         "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.WuilWP7i1fS2/output/models/epoch.030.pt"
    #     )
    # )
    if use_kv_cache:
        return export_model_kv_cached(model_config=model_config, checkpoint=checkpoint)
    return export_model_stateless(model_config=model_config, checkpoint=checkpoint)


@lru_cache
def _get_transformer_lm_base_config(layers: int, use_kv_cache: bool, use_gpu: bool) -> RasrConfig:
    onnx_model = _get_transformer_lm_onnx_model(layers=layers, use_kv_cache=use_kv_cache)
    vocab_file = get_lm_vocab(output_prefix="").vocab
    # vocab_file = tk.Path(
    #     "/work/asr4/berger/dependencies/librispeech/lm/kazuki_transformerlm_2019interspeech/vocabulary"
    # )

    if use_kv_cache:
        return get_lm_config_kv_cached(onnx_model=onnx_model, vocab_file=vocab_file, use_gpu=use_gpu)
    return get_lm_config_stateless(onnx_model=onnx_model, vocab_file=vocab_file, use_gpu=use_gpu)


def _get_transformer_lm_config(params: TransformerLmParams) -> RasrConfig:
    if params.onnx_model is None:
        base_config = _get_transformer_lm_base_config(
            layers=params.layers,
            use_kv_cache=params.use_kv_cache,
            use_gpu=params.use_gpu,
        )
    else:
        vocab_file = params.vocab_file if params.vocab_file is not None else get_lm_vocab(output_prefix="").vocab
        if params.use_kv_cache:
            base_config = get_lm_config_kv_cached(
                onnx_model=params.onnx_model,
                vocab_file=vocab_file,
                use_gpu=params.use_gpu,
            )
        else:
            base_config = get_lm_config_stateless(
                onnx_model=params.onnx_model,
                vocab_file=vocab_file,
                use_gpu=params.use_gpu,
            )
    config = _copy_with_scale(rasr_config=base_config, scale=params.scale)
    if params.max_batch_size is not None:
        if params.max_batch_size < 1:
            raise ValueError("LM max_batch_size must be positive")
        config.max_batch_size = params.max_batch_size
    return config


def get_word_transformer_onnx_model(params: TransformerLmParams) -> tk.Path:
    """Return the unmodified word-level Transformer ONNX model selected by ``params``."""

    if params.onnx_model is not None:
        return params.onnx_model
    return _get_transformer_lm_onnx_model(layers=params.layers, use_kv_cache=params.use_kv_cache)


@lru_cache
def _get_kazuki_trafo_base_config() -> RasrConfig:
    dependency_path = tk.Path("/work/asr4/berger/dependencies/librispeech/lm", hash_overwrite="DEPDENDENCY_LBS_LM")
    kazuki_transformer_path = dependency_path.join_right("kazuki_transformerlm_2019interspeech")

    config = RasrConfig()
    config.type = "simple-transformer"
    config.scale = 1.0
    config.vocab_file = kazuki_transformer_path.join_right("vocabulary")
    config.transform_output_negate = True
    config.vocab_unknown_word = "<UNK>"

    config.loader = RasrConfig()
    config.loader.type = "meta"
    config.loader.meta_graph_file = kazuki_transformer_path.join_right("inference.meta")
    config.loader.saved_model_file = Checkpoint(index_path=kazuki_transformer_path.join_right("network.030.index"))

    config.input_map = RasrConfig()
    config.input_map.info_0 = RasrConfig()
    config.input_map.info_0.param_name = "word"
    config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
    config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

    config.output_map = RasrConfig()
    config.output_map.info_0 = RasrConfig()
    config.output_map.info_0.param_name = "softmax"
    config.output_map.info_0.tensor_name = "output/output_batch_major"

    return config


def _get_kazuki_trafo_lm_config(params: KazukiTrafoLmParams) -> RasrConfig:
    base_config = _get_kazuki_trafo_base_config()
    return _copy_with_scale(rasr_config=base_config, scale=params.scale)


@lru_cache
def _get_bpe_lstm_lm_artifacts(bpe_size: int) -> Tuple[LstmLmConfig, PtCheckpoint]:
    model_config = lstm_lm_bpe.get_model_config(bpe_size=bpe_size)
    train_options = lstm_lm_bpe.get_train_options(bpe_size=bpe_size)

    # checkpoint = PtCheckpoint(
    #     tk.Path(
    #         "/work/asr4/rossenbach/sisyphus_work_folders/tts_decoder_asr_work/i6_core/returnn/training/ReturnnTrainingJob.EuWaxahLY8Ab/output/models/epoch.300.pt"
    #     )
    # )
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
