__all__ = [
    "get_arpa_lm_rasr_config",
    "get_nnlm_rasr_config",
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


def get_arpa_lm_rasr_config(
    filename: Union[str, Path], scale: float, image: Optional[Union[str, Path]] = None
):
    """
    :param filename: path to ARPA LM
    :param scale: LM scale
    :param image: global cache image
    :rtype: RasrConfig
    """
    config = rasr.RasrConfig()
    config.type = "ARPA"
    config.file = filename
    config.scale = scale

    if image is not None:
        config.image = image

    return config


def get_nnlm_rasr_config(
    lm_type: str,
    vocab_path: Union[Path, str],
    returnn_lm_inference_config: Union[returnn.ReturnnConfig, Path, str],
    returnn_checkpoint: Union[returnn.Checkpoint, str],
    scale: float = 1.0,
    *,
    vocab_unknown_word: str = "<UNK>",
    label_level: str = "word",
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
    """
    create RasrConfig for Neural LM
    two types are supported: tfrnn (LSTM LM) and simple-transformer (Trafo LM)

    :param vocab_path: LM vocabulary file
    :param returnn_lm_inference_config: LM RETURNN config, already prepared for inference
    :param returnn_checkpoint:
    :param vocab_unknown_word:
    :param label_level:
    :param scale:
    :param opt_batch_size:
    :param max_batch_size:
    :param min_batch_size:
    :param transform_output_negate:
    :param allow_reduced_history:
    :param native_op_args:
    :param tf_graph_args:
    :param returnn_python_exe:
    :param returnn_root:
    :rtype: RasrConfig
    """
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

    if isinstance(returnn_checkpoint, str):
        assert returnn_checkpoint.endswith(".index")
        returnn_checkpoint = returnn.Checkpoint(returnn_checkpoint)

    config = rasr.RasrConfig()
    config.type = lm_type
    config.scale = scale

    config.vocab_file = vocab_path
    config.vocab_unknown_word = vocab_unknown_word

    config.transform_output_negate = transform_output_negate
    config.allow_reduced_history = allow_reduced_history

    config.max_batch_size = max_batch_size
    config.min_batch_size = min_batch_size
    config.opt_batch_size = opt_batch_size

    config.input_map.info_0.param_name = label_level
    config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
    config.input_map.info_0.seq_length_tensor_name = (
        "extern_data/placeholders/delayed/delayed_dim0_size"
    )

    config.loader.type = "meta"
    config.loader.required_libraries = returnn.CompileNativeOpJob(
        "NativeLstm2", **native_op_args
    ).out_op
    config.loader.meta_graph_file = returnn.CompileTFGraphJob(
        returnn_lm_inference_config, **tf_graph_args
    ).out_graph
    config.loader.saved_model_file = returnn_checkpoint.ckpt_path

    config.output_map.info_0.param_name = "softmax"
    config.output_map.info_0.tensor_name = "output/output_batch_major"

    return config


def add_lm_rasr_config_to_crp(
    crp: rasr.CommonRasrParameters,
    lm_args: Dict,
    *,
    lm_lookahead_args: Optional[Dict] = None,
    returnn_python_exe: Optional[Union[str, Path]] = None,
    returnn_root: Optional[Union[str, Path]] = None,
):
    """
    adds a LM (ngram or NN) to a crp
    :param crp: CommonRasrParameters to add LM to
    :param lm_args: LM params (ngram, LSTM, Trafo)
    :param lm_lookahead_args: use a different LM for lm lookahead
    :param returnn_python_exe:
    :param returnn_root:
    """
    lm_type = lm_args.pop("type").lower()
    if lm_type == "arpa":
        crp.language_model_config = get_arpa_lm_rasr_config(**lm_args)
    elif lm_type == "tfrnn" or lm_type == "simple-transformer":
        lm_args["returnn_python_exe"] = returnn_python_exe
        lm_args["returnn_root"] = returnn_root
        crp.language_model_config = get_nnlm_rasr_config(**lm_args)
    else:
        raise NotImplementedError

    if lm_lookahead_args is not None:
        crp.recognizer_config.lm_lookahead = get_arpa_lm_rasr_config(
            **lm_lookahead_args
        )
