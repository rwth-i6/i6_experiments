"""
Contains the serializers for the ReturnnConfig epilog which write the model import and task functions import
including serializing their parameters.
"""
from typing import Any, Dict, Optional

from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from i6_core.util import instanciate_delayed
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import ExternalImport, PartialImport, NonhashedCode
from . import PACKAGE
from .default_tools import I6_NATIVE_OPS_REPO_PATH


def serialize_extern_data(extern_data: Dict[str, Any]):
    from returnn.util.pprint import pformat

    extern_data_str = f"extern_data = {pformat(extern_data)}\n"

    return NonhashedCode(extern_data_str)


def serialize_training(
    network_module: str,
    train_step_module: str,
    net_args: Dict[str, Any],
    train_args: Dict[str, Any],
    extern_data: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
    debug: bool = False,
) -> Collection:
    """
    Helper function to create the serialization collection

    :param network_module: path to the pytorch config file containing Model
    :param net_args: arguments for the model
    :param unhashed_net_args: as above but not hashed
    :param include_native_ops: include the i6_native_ops for e.g. Fast-Baum-Welch or Warp-RNNT
    :param debug: run training in debug mode: linking from recipe instead of copy
    :return: Collection object to be added to the ReturnnConfig epilog
    """
    package = PACKAGE

    pytorch_model_import = PartialImport(
        code_object_path=package + f".{network_module}.Model",
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )
    pytorch_train_step_import = PartialImport(
        code_object_path=package + f".{train_step_module}.train_step",
        unhashed_package_root=PACKAGE,
        hashed_arguments=train_args,
        unhashed_arguments={},
    )
    # i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)

    serializer_objects = [
        # i6_models,
        serialize_extern_data(instanciate_delayed(extern_data)),
        pytorch_model_import,
        pytorch_train_step_import,
    ]

    if include_native_ops:
        i6_native_ops = ExternalImport(import_path=I6_NATIVE_OPS_REPO_PATH)
        # has to be in the beginning
        serializer_objects.insert(0, i6_native_ops)

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer


def serialize_forward(
    network_module: str,
    net_args: Dict[str, Any],
    extern_data: Dict[str, Any],
    vocab_opts: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    forward_module: Optional[str] = None,
    forward_step_name: str = "forward_step",
    forward_init_args: Optional[Dict[str, Any]] = None,
    unhashed_forward_init_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
    debug: bool = False,
):
    """
    Serialize for a forward job. Can be used e.g. for search or prior computation.

    :param network_module: path to the pytorch config file containing Model
    :param net_args: arguments for the model
    :param unhashed_net_args: as above but not hashed
    :param forward_module: optionally define a module file which contains the forward definition.
        If not provided the network_module is used.
    :param forward_step_name: path to the search decoder file containing forward_step and hooks
    :param forward_init_args: additional arguments to pass to forward_init
    :param unhashed_forward_init_args: additional non-hashed arguments to pass to forward_init
    :param include_native_ops: include the i6_native_ops for e.g. Fast-Baum-Welch or Warp-RNNT
    :param debug: run training in debug mode: linking from recipe instead of copy
    :return:
    """

    package = PACKAGE  # + ".pytorch_networks"

    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )

    # i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)

    serializer_objects = [
        # i6_models,
        serialize_extern_data(instanciate_delayed(extern_data)),
        pytorch_model_import,
    ]

    if include_native_ops:
        i6_native_ops = ExternalImport(import_path=I6_NATIVE_OPS_REPO_PATH)
        # has to be in the beginning
        serializer_objects.insert(0, i6_native_ops)

    forward_module = forward_module or network_module

    decoder_params = {
        "beam_size": 12,
        "max_tokens_per_sec": 20,
        "sample_rate": 16_000,
    }

    if forward_step_name == "forward_step_ctc_decoding_v2":
        decoder_params["ctc_scale"] = 1.0
        decoder_params["sllm_scale"] = 1.0
        decoder_params["lm_scale"] = 0.0
        decoder_params["prior_scale"] = 0.0

        if "ctc_scale" in forward_init_args and "sllm_scale" in forward_init_args:
            decoder_params["ctc_scale"] = forward_init_args["ctc_scale"]
            decoder_params["sllm_scale"] = forward_init_args["sllm_scale"]


    forward_step = PartialImport(
        code_object_path=package + ".%s.forward_step.%s" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        import_as="forward_step",
        hashed_arguments=decoder_params,
        unhashed_arguments={},
    )

    assert vocab_opts["class"] == "SentencePieces"
    spm_model_file = vocab_opts["model_file"]
    vocab_file = ExtractSentencePieceVocabJob(model=spm_model_file).out_vocab
    callback = PartialImport(
        code_object_path=f"{package}.recognition.aed.callback.RecognitionToTextDictCallback",
        import_as="forward_callback",
        hashed_arguments={"vocab": vocab_file},
        unhashed_arguments={},
        unhashed_package_root=None,
    )
    # init_hook = PartialImport(
    #     code_object_path=package + ".%s.%s_init_hook" % (forward_module, forward_step_name),
    #     unhashed_package_root=PACKAGE,
    #     hashed_arguments=forward_init_args or {},
    #     unhashed_arguments=unhashed_forward_init_args or {},
    #     import_as="forward_init_hook",
    # )
    # finish_hook = Import(
    #     code_object_path=package + ".%s.%s_finish_hook" % (forward_module, forward_step_name),
    #     unhashed_package_root=PACKAGE,
    #     import_as="forward_finish_hook",
    # )
    serializer_objects.extend([forward_step, callback])

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer

def serialize_forward_v2(
        network_import_path: str,
        net_args: Dict[str, Any],
        extern_data: Dict[str, Any],
        vocab_opts: Dict[str, Any],
        unhashed_net_args: Optional[Dict[str, Any]] = None,
        forward_module: Optional[str] = None,
        forward_step_name: str = "forward_step",
        callback_name: str = "RecognitionToTextDictCallback",
        include_native_ops=False,
        debug: bool = False,
        forward_method: Optional[str] = None,
        forward_args: Optional[Dict[str, Any]] = None,
        callback_opts: Optional[Dict[str, Any]] = None,
):
    """
    Serialize for a forward job. Can be used e.g. for search or prior computation.

    :param network_import_path: path to the pytorch config file containing Model
    :param net_args: arguments for the model
    :param unhashed_net_args: as above but not hashed
    :param forward_module: optionally define a module file which contains the forward definition.
        If not provided the network_module is used.
    :param forward_step_name: path to the search decoder file containing forward_step and hooks
    :param include_native_ops: include the i6_native_ops for e.g. Fast-Baum-Welch or Warp-RNNT
    :param debug: run training in debug mode: linking from recipe instead of copy
    :return:
    """
    if forward_module is None:
        forward_module = network_import_path
    if forward_args is None:
        forward_args = {}

    pytorch_model_import = PartialImport(
        code_object_path=f"{PACKAGE}.{network_import_path}",
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )

    serializer_objects = [
        serialize_extern_data(instanciate_delayed(extern_data)),
        pytorch_model_import,
    ]

    if include_native_ops:
        i6_native_ops = ExternalImport(import_path=I6_NATIVE_OPS_REPO_PATH)
        # has to be in the beginning
        serializer_objects.insert(0, i6_native_ops)

    if forward_method is None:
        forward_object_path = f"{PACKAGE}.{forward_module}.{forward_step_name}"  # For backward compatibility
    else:
        forward_object_path = f"{PACKAGE}.{forward_module}.{forward_step_name}.{forward_method}"

    forward_step = PartialImport(
        code_object_path=forward_object_path,
        unhashed_package_root=PACKAGE,
        import_as="forward_step",
        hashed_arguments=forward_args,
        unhashed_arguments={},
    )

    hashed_arguments = {}
    if callback_name == "RecognitionToTextDictCallback" or callback_name == "RecognitionToTextDictCallbackV2":
        assert vocab_opts["class"] == "SentencePieces"
        spm_model_file = vocab_opts["model_file"]
        vocab_file = ExtractSentencePieceVocabJob(model=spm_model_file).out_vocab
        hashed_arguments = {"vocab": vocab_file}

    if callback_opts is not None:
        hashed_arguments.update(callback_opts)

    callback = PartialImport(
        code_object_path=f"{PACKAGE}.{forward_module}.callback.{callback_name}",
        import_as="forward_callback",
        hashed_arguments=hashed_arguments,
        unhashed_arguments={},
        unhashed_package_root=None,
    )

    serializer_objects.extend([forward_step, callback])

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={PACKAGE},
    )
    return serializer