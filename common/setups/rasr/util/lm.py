__all__ = [
    "get_arpa_lm_rasr_config",
    "get_tfrnn_lm_rasr_config",
    "add_lm_rasr_config_to_crp",
]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

# -------------------- Init --------------------

Path = tk.setup_path(__package__)


# -------------------- LM Rasr Config --------------------


def get_arpa_lm_rasr_config(filename: Union[str, Path], scale: float):
    """
    :param filename: path to ARPA LM
    :param scale: LM scale
    :rtype: RasrConfig
    """
    config = rasr.RasrConfig()
    config.type = "ARPA"
    config.file = filename
    config.scale = scale
    return config


def get_tfrnn_lm_rasr_config(
    vocab_path: Union[Path, str],
    returnn_lm_inference_config: Union[returnn.ReturnnConfig, Path, str],
    returnn_checkpoint: Union[returnn.Checkpoint, str],
    vocab_unknown_word: str = "<UNK>",
    label_level: str = "word",
    scale: float = 1.0,
    *,
    opt_batch_size: int = 64,
    max_batch_size: int = 128,
    min_batch_size: int = 4,
    transform_output_negate: bool = True,
    allow_reduced_history: bool = True,
    native_op_args: Optional[Dict] = None,
    tf_graph_args: Optional[Dict] = None,
    returnn_python_exe: Optional[Union[Path, str]] = None,
    returnn_root: Optional[Union[Path, str]] = None,
):
    if native_op_args is None:
        native_op_args = {}
    if tf_graph_args is None:
        tf_graph_args = {}

    if not hasattr(native_op_args, "returnn_python_exe"):
        native_op_args["returnn_python_exe"] = returnn_python_exe
    if not hasattr(native_op_args, "returnn_root"):
        native_op_args["returnn_root"] = returnn_root

    if not hasattr(tf_graph_args, "returnn_python_exe"):
        tf_graph_args["returnn_python_exe"] = returnn_python_exe
    if not hasattr(tf_graph_args, "returnn_root"):
        tf_graph_args["returnn_root"] = returnn_root

    config = rasr.RasrConfig()
    config.type = "tfrnn"
    config.scale = scale

    config.vocab_file = vocab_path  # /u/zhou/asr-exps/librispeech/dependencies/kazuki_lstmlm_27062019/vocabulary
    config.vocab_unknown_word = vocab_unknown_word  # <UNK>

    config.transform_output_negate = transform_output_negate  # yes
    config.allow_reduced_history = allow_reduced_history  # yes

    config.max_batch_size = max_batch_size  # 128
    config.min_batch_size = min_batch_size  # 4
    config.opt_batch_size = opt_batch_size  # 64

    config.input_map.info_0.param_name = label_level  # word
    config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"  # extern_data/placeholders/delayed/delayed
    config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"  # extern_data/placeholders/delayed/delayed_dim0_size

    config.loader.type = "meta"  # meta
    config.loader.required_libraries = returnn.CompileNativeOpJob(
        "NativeLstm2", **native_op_args
    ).out_op  # /u/zhou/libs/nativelstm2/tf1.12/NativeLstm2.so
    config.loader.meta_graph_file = returnn.CompileTFGraphJob(
        returnn_lm_inference_config, **tf_graph_args
    ).out_graph  # /u/luescher/setups/librispeech/2019-07-23--data-augmentation/work/crnn/compile/CompileTFGraphJob.0dxq1DSvOxuN/output/graph.meta
    config.loader.saved_model_file = (
        returnn_checkpoint.ckpt_path
    )  # /u/zhou/asr-exps/librispeech/dependencies/kazuki_lstmlm_27062019/network.040

    config.output_map.info_0.param_name = "softmax"  # softmax
    config.output_map.info_0.tensor_name = (
        "output/output_batch_major"  # output/output_batch_major
    )

    return config


def add_lm_rasr_config_to_crp(crp, lm_args, returnn_python_exe, returnn_root):
    lm_type = lm_args.pop("type").lower()
    if lm_type == "arpa":
        crp.language_model_config = get_arpa_lm_rasr_config(**lm_args)
    elif lm_type == "tfrnn":
        lm_args["returnn_python_exe"] = returnn_python_exe
        lm_args["returnn_root"] = returnn_root
        crp.language_model_config = get_tfrnn_lm_rasr_config(**lm_args)
    else:
        raise NotImplementedError
