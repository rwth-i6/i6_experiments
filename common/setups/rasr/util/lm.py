__all__ = [
    "get_arpa_lm_rasr_config",
    "get_nnlm_rasr_config",
    "add_lm_rasr_config_to_crp",
]

from typing import Dict, List, Optional, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.users.luescher.setups.rasr.config.lm_config as lm_config

# -------------------- Init --------------------

Path = tk.setup_path(__package__)


# -------------------- LM Rasr Config --------------------


def get_arpa_lm_rasr_config(
    lm_path: tk.Path, scale: float, image: Optional[tk.Path] = None
):
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
    vocab_path: Union[tk.Path, str],
    returnn_lm_inference_config: Union[returnn.ReturnnConfig, tk.Path, str],
    returnn_checkpoint: returnn.Checkpoint,
    *,
    scale: Optional[float] = None,
    unknown_symbol: str = "<UNK>",
    output_layer_type: str = "softmax",
    transform_output_negate: bool = True,
    native_op_args: Optional[Dict] = None,
    tf_graph_args: Optional[Dict] = None,
    returnn_python_exe: Optional[Union[tk.Path, str]] = None,
    returnn_root: Optional[Union[tk.Path, str]] = None,
    libraries: Optional[Union[tk.Path, List[tk.Path]]] = None,
    **kwargs,
):
    """
    create RasrConfig for Neural LM
    two types are supported: tfrnn (LSTM/Trafo LM) and simple-tf-neural (Trafo LM)

    :param lm_type: which kind of LM to use
    :param vocab_path: LM vocabulary file
    :param returnn_lm_inference_config: LM RETURNN config, already prepared for inference
    :param returnn_checkpoint:
    :param scale:
    :param unknown_symbol:
    :param output_layer_type:
    :param transform_output_negate:
    :param native_op_args:
    :param tf_graph_args:
    :param returnn_python_exe:
    :param returnn_root:
    :param libraries:
    :rtype: RasrConfig
    """
    if isinstance(returnn_checkpoint, str):
        assert returnn_checkpoint.endswith(".index")
        returnn_checkpoint = returnn.Checkpoint(returnn_checkpoint)

    if isinstance(libraries, tk.Path):
        libraries = [libraries]

    if libraries is None:
        libraries = []

    if tf_graph_args is None:
        tf_graph_args = {}
    if not hasattr(tf_graph_args, "returnn_python_exe"):
        tf_graph_args["returnn_python_exe"] = returnn_python_exe
    if not hasattr(tf_graph_args, "returnn_root"):
        tf_graph_args["returnn_root"] = returnn_root
    meta_graph_path = returnn.CompileTFGraphJob(
        returnn_lm_inference_config, **tf_graph_args
    ).out_graph

    if native_op_args is None:
        native_op_args = {}
    if not hasattr(native_op_args, "returnn_python_exe"):
        native_op_args["returnn_python_exe"] = returnn_python_exe
    if not hasattr(native_op_args, "returnn_root"):
        native_op_args["returnn_root"] = returnn_root
    libraries.append(returnn.CompileNativeOpJob("NativeLstm2", **native_op_args).out_op)

    if lm_type == "tfrnn":
        state_manager = kwargs.pop("state_manager")

        config = lm_config.TfRnnLmRasrConfig(
            vocab_path=vocab_path,
            meta_graph_path=meta_graph_path,
            returnn_checkpoint=returnn_checkpoint,
            scale=scale,
            unknown_symbol=unknown_symbol,
            transform_output_negate=transform_output_negate,
            output_layer_type=output_layer_type,
            libraries=libraries,
            state_manager=state_manager,
            **kwargs,
        )
    elif lm_type == "simple-tf-neural":
        config = lm_config.SimpleTfNeuralLmRasrConfig(
            vocab_path=vocab_path,
            meta_graph_path=meta_graph_path,
            returnn_checkpoint=returnn_checkpoint,
            scale=scale,
            unknown_symbol=unknown_symbol,
            transform_output_negate=transform_output_negate,
            output_layer_type=output_layer_type,
            libraries=libraries,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return config.get()


def add_lm_rasr_config_to_crp(
    crp: rasr.CommonRasrParameters,
    lm_args: Dict,
    *,
    lm_lookahead_args: Optional[Dict] = None,
    returnn_python_exe: Optional[Union[str, tk.Path]] = None,
    returnn_root: Optional[Union[str, tk.Path]] = None,
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
