from dataclasses import dataclass
from typing import Union

from i6_core.lm.lm_image import CreateLmImageJob
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters, crp_add_default_output
from i6_core.returnn import Checkpoint, PtCheckpoint
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_experiments.common.datasets.librispeech.vocab import get_lm_vocab
from sisyphus import tk

from ...experiments.librispeech.training import bpe_lstm_lm, bpe_transformer_lm, word_transformer_lm
from ...model_pipelines.lstm_lm.label_scorer_config import get_lstm_lm_label_scorer_config
from ...model_pipelines.transformer_lm.export import export_model_kv_cached, export_model_stateless
from ...model_pipelines.transformer_lm.label_scorer_config import (
    get_bpe_transformer_lm_label_scorer_config,
)
from ...model_pipelines.transformer_lm.lm_config import get_lm_config_kv_cached, get_lm_config_stateless
from ...tools import rasr_binary_path


@dataclass
class ArpaLmParams:
    scale: float = 1.0


def _get_arpa_lm_config(lexicon_file: tk.Path, params: ArpaLmParams) -> RasrConfig:
    arpa_lm = get_arpa_lm_dict()["4gram"]
    rasr_config = RasrConfig()
    rasr_config.type = "ARPA"
    rasr_config.file = arpa_lm
    rasr_config.scale = params.scale

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = rasr_binary_path.join_right("lm-util.linux-x86_64-standard")  # type: ignore
    crp.language_model_config = rasr_config  # type: ignore
    crp.lexicon_config = RasrConfig()  # type: ignore
    crp.lexicon_config.file = lexicon_file  # type: ignore
    rasr_config.image = CreateLmImageJob(crp, mem=8).out_image

    return rasr_config


@dataclass
class TransformerLmParams:
    scale: float = 1.0
    layers: int = 96
    use_kv_cache: bool = True  # Generally, kv cache is slower on GPU because setting up the cache inputs takes so long
    use_gpu: bool = False


def _get_transformer_lm_config(params: TransformerLmParams) -> RasrConfig:
    train_options = word_transformer_lm.get_train_options()
    train_options.register_outputs = False
    model_config = word_transformer_lm.get_model_config()
    model_config.num_layers = params.layers
    word_transformer_model = word_transformer_lm.run(train_options=train_options, model_config=model_config)
    vocab_file = get_lm_vocab(output_prefix="").vocab
    # checkpoint = PtCheckpoint(
    #     tk.Path(
    #         "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.WuilWP7i1fS2/output/models/epoch.030.pt"
    #     )
    # )
    if params.use_kv_cache:
        onnx_model = export_model_kv_cached(
            model_config=word_transformer_model.model_config, checkpoint=word_transformer_model.get_checkpoint()
        )
    else:
        onnx_model = export_model_stateless(
            model_config=word_transformer_model.model_config, checkpoint=word_transformer_model.get_checkpoint()
        )
    # vocab_file = tk.Path(
    #     "/work/asr4/berger/dependencies/librispeech/lm/kazuki_transformerlm_2019interspeech/vocabulary"
    # )

    if params.use_kv_cache:
        return get_lm_config_kv_cached(
            onnx_model=onnx_model, vocab_file=vocab_file, lm_scale=params.scale, use_gpu=params.use_gpu
        )
    else:
        return get_lm_config_stateless(
            onnx_model=onnx_model, vocab_file=vocab_file, lm_scale=params.scale, use_gpu=params.use_gpu
        )


@dataclass
class KazukiTrafoLmParams:
    scale: float = 1.0


def _get_kazuki_trafo_lm_config(params: KazukiTrafoLmParams) -> RasrConfig:
    dependency_path = tk.Path("/work/asr4/berger/dependencies/librispeech/lm", hash_overwrite="DEPDENDENCY_LBS_LM")
    kazuki_transformer_path = dependency_path.join_right("kazuki_transformerlm_2019interspeech")

    config = RasrConfig()
    config.type = "simple-transformer"
    config.scale = params.scale
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


WordLmParams = Union[ArpaLmParams, TransformerLmParams, KazukiTrafoLmParams]


def get_word_lm_config(lexicon_file: tk.Path, params: WordLmParams) -> RasrConfig:
    if isinstance(params, ArpaLmParams):
        return _get_arpa_lm_config(lexicon_file=lexicon_file, params=params)
    if isinstance(params, TransformerLmParams):
        return _get_transformer_lm_config(params=params)
    if isinstance(params, KazukiTrafoLmParams):
        return _get_kazuki_trafo_lm_config(params=params)


def get_bpe_lstm_label_scorer_config(bpe_size: int = 128, use_gpu: bool = False, scale: float = 1.0) -> RasrConfig:
    assert bpe_size == 128
    model_config = bpe_lstm_lm.get_model_config(bpe_size=bpe_size)

    lstm_lm_checkpoint = PtCheckpoint(
        tk.Path(
            "/work/asr4/rossenbach/sisyphus_work_folders/tts_decoder_asr_work/i6_core/returnn/training/ReturnnTrainingJob.EuWaxahLY8Ab/output/models/epoch.300.pt"
        )
    )

    return get_lstm_lm_label_scorer_config(
        model_config=model_config,
        checkpoint=lstm_lm_checkpoint,
        scale=scale,
        execution_provider_type="cuda" if use_gpu else None,
    )


def get_bpe_transformer_label_scorer_config(
    bpe_size: int = 128, num_layers: int = 96, use_gpu: bool = False, scale: float = 1.0
) -> RasrConfig:
    model_config = bpe_transformer_lm.get_model_config(bpe_size=bpe_size)
    model_config.num_layers = num_layers

    train_options = bpe_transformer_lm.get_train_options(bpe_size=bpe_size)
    train_options.register_outputs = False

    trafo_lm = bpe_transformer_lm.run(model_config=model_config, train_options=train_options)

    return get_bpe_transformer_lm_label_scorer_config(
        model_config=trafo_lm.model_config,
        checkpoint=trafo_lm.get_checkpoint(),
        scale=scale,
        execution_provider_type="cuda" if use_gpu else None,
    )
