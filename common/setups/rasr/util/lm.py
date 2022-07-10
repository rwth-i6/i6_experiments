__all__ = [
    "get_arpa_lm_rasr_config",
    "get_nnlm_rasr_config",
    "add_lm_rasr_config_to_crp",
]

from typing import Dict, Optional, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.config.lm_config as lm_config

# -------------------- Init --------------------

Path = tk.setup_path(__package__)


# -------------------- LM Rasr Config --------------------


def get_arpa_lm_rasr_config(lm_path: Path, scale: float, image: Optional[Path] = None):
    """
    :param lm_path: path to ARPA LM
    :param scale: LM scale
    :param image: global cache image
    :rtype: RasrConfig
    """

    config = lm_config.ArpaLmRasrConfig(lm_path=lm_path, scale=scale, image=image)

    return config.get()


def get_nnlm_rasr_config(
    lm_type: str,
    vocab_path: Union[Path, str],
    returnn_lm_inference_config: Union[returnn.ReturnnConfig, Path, str],
    returnn_checkpoint: returnn.Checkpoint,
    *,
    scale: Optional[float] = None,
    vocab_unknown_word: str = "<UNK>",
    label_level: str = "word",
    max_batch_size: int = 128,
    transform_output_negate: bool = True,
    native_op_args: Optional[Dict] = None,
    tf_graph_args: Optional[Dict] = None,
    returnn_python_exe: Optional[Union[Path, str]] = None,
    returnn_root: Optional[Union[Path, str]] = None,
):
    """
    create RasrConfig for Neural LM
    two types are supported: tfrnn (LSTM/Trafo LM) and simple-tf-neural (Trafo LM)

    :param lm_type: which kind of LM to use
    :param vocab_path: LM vocabulary file
    :param returnn_lm_inference_config: LM RETURNN config, already prepared for inference
    :param returnn_checkpoint:
    :param vocab_unknown_word:
    :param label_level:
    :param scale:
    :param max_batch_size:
    :param transform_output_negate:
    :param native_op_args:
    :param tf_graph_args:
    :param returnn_python_exe:
    :param returnn_root:
    :rtype: RasrConfig
    """
    kwargs = locals()
    del kwargs["lm_type"]
    del kwargs["native_op_args"]
    del kwargs["tf_graph_args"]
    del kwargs["returnn_python_exe"]
    del kwargs["returnn_root"]

    rasr_conf_class = (
        lm_config.TfRnnLmRasrConfig
        if lm_type == "tfrnn"
        else lm_config.SimpleTfNeuralLmRasrConfig
    )

    if isinstance(returnn_checkpoint, str):
        assert returnn_checkpoint.endswith(".index")
        returnn_checkpoint = returnn.Checkpoint(returnn_checkpoint)

    if tf_graph_args is not None:
        if not hasattr(tf_graph_args, "returnn_python_exe"):
            tf_graph_args["returnn_python_exe"] = returnn_python_exe
        if not hasattr(tf_graph_args, "returnn_root"):
            tf_graph_args["returnn_root"] = returnn_root
        kwargs["meta_graph_path"] = returnn.CompileTFGraphJob(
            returnn_lm_inference_config, **tf_graph_args
        ).out_graph

    if native_op_args is not None:
        if not hasattr(native_op_args, "returnn_python_exe"):
            native_op_args["returnn_python_exe"] = returnn_python_exe
        if not hasattr(native_op_args, "returnn_root"):
            native_op_args["returnn_root"] = returnn_root
        kwargs["libraries"] = returnn.CompileNativeOpJob(
            "NativeLstm2", **native_op_args
        ).out_op

    config = rasr_conf_class(**kwargs)

    return config.get()


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
    elif lm_type == "tfrnn" or lm_type == "simple-tf-neural":
        lm_args["returnn_python_exe"] = returnn_python_exe
        lm_args["returnn_root"] = returnn_root
        crp.language_model_config = get_nnlm_rasr_config(lm_type=lm_type, **lm_args)
    else:
        raise NotImplementedError

    if lm_lookahead_args is not None:
        if crp.recognizer_config is None:
            crp.recognizer_config = rasr.RasrConfig()
        crp.recognizer_config.lm_lookahead = get_arpa_lm_rasr_config(
            **lm_lookahead_args
        )
