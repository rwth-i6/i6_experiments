"""
Contains the serializers for the ReturnnConfig epilog which write the model import and task functions import
including serializing their parameters.
"""
from typing import Any, Dict, Optional

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import ExternalImport, Import, PartialImport

from . import PACKAGE
from .default_tools import I6_MODELS_REPO_PATH, I6_NATIVE_OPS_REPO_PATH, RECIPE_ROOT


def serialize_training(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    train_step_args: Optional[Dict[str, Any]] = None,
    unhashed_train_step_args: Optional[Dict[str, Any]] = None,
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
    package = PACKAGE + ".pytorch_networks"

    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )
    pytorch_train_step = PartialImport(
        code_object_path=package + ".%s.train_step" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=train_step_args or {},
        unhashed_arguments=unhashed_train_step_args or {},
        import_as="train_step",
    )
    i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)

    serializer_objects = [
        i6_models,
        pytorch_model_import,
        pytorch_train_step,
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
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    forward_module: Optional[str] = None,
    forward_step_name: str = "forward",
    forward_callback_name: str = "ForwardCallback",
    forward_init_args: Optional[Dict[str, Any]] = None,
    unhashed_forward_init_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
    debug: bool = False,
):
    """
    Serialize for a forward job. Can be used e.g. for search or prior computation.

    Real RETURNN drives the forward task with ``forward_step`` (marks the model outputs)
    plus a ``forward_callback`` (a :class:`returnn.forward_iface.ForwardCallbackIface`
    that does the per-sequence post-processing). The decoder/prior module therefore has to
    define a ``<forward_step_name>_step`` function and a ``<forward_callback_name>`` class.

    :param network_module: path to the pytorch config file containing Model
    :param net_args: arguments for the model
    :param unhashed_net_args: as above but not hashed
    :param forward_module: optionally define a module file which contains the forward definition.
        If not provided the network_module is used.
    :param forward_step_name: prefix of the ``forward_step`` function in the module (``<name>_step``)
    :param forward_callback_name: name of the ForwardCallbackIface subclass in the module
    :param forward_init_args: arguments passed to the callback constructor (hashed)
    :param unhashed_forward_init_args: as above but not hashed
    :param include_native_ops: include the i6_native_ops for e.g. Fast-Baum-Welch or Warp-RNNT
    :param debug: run training in debug mode: linking from recipe instead of copy
    :return:
    """

    package = PACKAGE + ".pytorch_networks"

    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )

    i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)
    recipe_root = ExternalImport(import_path=RECIPE_ROOT)

    serializer_objects = [
        i6_models,
        recipe_root,
        pytorch_model_import,
    ]

    if include_native_ops:
        i6_native_ops = ExternalImport(import_path=I6_NATIVE_OPS_REPO_PATH)
        # has to be in the beginning
        serializer_objects.insert(0, i6_native_ops)

    forward_module = forward_module or network_module

    forward_step = Import(
        code_object_path=package + ".%s.%s_step" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        import_as="forward_step",
    )
    forward_callback = PartialImport(
        code_object_path=package + ".%s.%s" % (forward_module, forward_callback_name),
        unhashed_package_root=PACKAGE,
        hashed_arguments=forward_init_args or {},
        unhashed_arguments=unhashed_forward_init_args or {},
        import_as="forward_callback",
    )
    serializer_objects.extend([forward_step, forward_callback])

    serializer = Collection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer
