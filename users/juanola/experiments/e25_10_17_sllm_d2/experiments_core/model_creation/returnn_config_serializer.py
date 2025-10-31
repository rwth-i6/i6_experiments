"""
Contains the serializers for the ReturnnConfig epilog which write the model import and task functions import
including serializing their parameters.
"""
from typing import Any, Dict, Optional

from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob
from i6_core.util import instanciate_delayed
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import ExternalImport, PartialImport, NonhashedCode
from ... import ROOT_PACKAGE
from ...default_tools import I6_NATIVE_OPS_REPO_PATH


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
    pytorch_model_import = PartialImport(  # TODO: path is not exactly right! it adds "recipe"!!
        code_object_path=f"{ROOT_PACKAGE}.{network_module}.Model",  # TODO: extract Model name!!
        unhashed_package_root=ROOT_PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )
    pytorch_train_step_import = PartialImport(
        code_object_path=f"{ROOT_PACKAGE}.{train_step_module}.train_step", # TODO: conventions!
        unhashed_package_root=ROOT_PACKAGE,
        hashed_arguments=train_args,
        unhashed_arguments={},
    )

    serializer_objects = [
        # i6_models,
        serialize_extern_data(instanciate_delayed(extern_data)),
        pytorch_model_import,
        pytorch_train_step_import,
    ]

    if include_native_ops:
        i6_native_ops = ExternalImport(import_path=I6_NATIVE_OPS_REPO_PATH)
        serializer_objects.insert(0, i6_native_ops)  # has to be in the beginning

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            ROOT_PACKAGE,
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
        forward_step_name: str = "forward",
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

    pytorch_model_import = PartialImport(
        code_object_path=f"{ROOT_PACKAGE}.{network_module}.Model",
        unhashed_package_root=ROOT_PACKAGE,
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

    forward_module = forward_module or network_module

    forward_step = PartialImport(
        code_object_path=f"{ROOT_PACKAGE}.{forward_module}.{forward_step_name}_step", # TODO: this _step is not sane
        unhashed_package_root=ROOT_PACKAGE,
        import_as="forward_step",
        hashed_arguments={
            "beam_size": 12,
            "max_tokens_per_sec": 20,
            "sample_rate": 16_000,
        },
        unhashed_arguments={},
    )

    assert vocab_opts["class"] == "SentencePieces"
    spm_model_file = vocab_opts["model_file"]
    vocab_file = ExtractSentencePieceVocabJob(model=spm_model_file).out_vocab
    callback = PartialImport(
        code_object_path=f"{ROOT_PACKAGE}.recognition.callback.RecognitionToTextDictCallback",
        import_as="forward_callback",
        hashed_arguments={"vocab": vocab_file},
        unhashed_arguments={},
        unhashed_package_root=None,
    )

    serializer_objects.extend([forward_step, callback])

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={ROOT_PACKAGE},
    )
    return serializer
