"""
Contains the serializers for the ReturnnConfig epilog which write the model import and task functions import
including serializing their parameters.
"""

from typing import Any, Dict, Optional

from i6_core.util import instanciate_delayed

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import ExternalImport, Import, PartialImport, NonhashedCode

from ..models import PACKAGE


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
    train_step_import_as: Optional[str] = None,
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
        code_object_path=f"{package}.{network_module}",
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )
    pytorch_train_step = PartialImport(
        code_object_path=f"{package}.{train_step_module}",
        unhashed_package_root=PACKAGE,
        hashed_arguments=train_args,
        unhashed_arguments={},
        import_as=train_step_import_as,  # this changes the hash
    )

    serializer_objects = [
        serialize_extern_data(instanciate_delayed(extern_data)),
        pytorch_model_import,
        pytorch_train_step,
    ]

    serializer = Collection(
        serializer_objects=serializer_objects,
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
    callback_module: Optional[str] = None,
    callback_opts: Optional[Dict[str, Any]] = None,
    forward_init_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
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
        code_object_path=f"{PACKAGE}.{network_module}",
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )

    serializer_objects = [
        serialize_extern_data(instanciate_delayed(extern_data)),
        pytorch_model_import,
    ]

    forward_step = PartialImport(
        code_object_path=f"{PACKAGE}.{forward_module}",
        unhashed_package_root=PACKAGE,
        import_as="forward_step",
        hashed_arguments=forward_init_args,
        unhashed_arguments={},
    )

    if callback_opts is None:
        callback_opts = {}
    callback_opts["vocab"] = vocab_opts
    callback = PartialImport(
        code_object_path=f"{PACKAGE}.{callback_module}",
        import_as="forward_callback",
        hashed_arguments=callback_opts,
        unhashed_arguments={},
        unhashed_package_root=None,
    )
    serializer_objects.extend([forward_step, callback])

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=False,
        packages={
            PACKAGE,
        },
    )
    return serializer
