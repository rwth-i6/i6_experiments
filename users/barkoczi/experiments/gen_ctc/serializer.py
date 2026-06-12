from typing import Any, Dict, Optional

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import ExternalImport, Import, PartialImport

from . import PACKAGE
from .default_tools import I6_MODELS_REPO_PATH, I6_NATIVE_OPS_REPO_PATH


def serialize_training(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
    debug: bool = False,
) -> Collection:
    package = PACKAGE + ".pytorch_networks"
    serializer_objects = [
        ExternalImport(import_path=I6_MODELS_REPO_PATH),
        PartialImport(
            code_object_path=package + ".%s.Model" % network_module,
            unhashed_package_root=PACKAGE,
            hashed_arguments=net_args,
            unhashed_arguments=unhashed_net_args or {},
            import_as="get_model",
        ),
        Import(code_object_path=package + ".%s.train_step" % network_module, unhashed_package_root=PACKAGE),
    ]
    if include_native_ops:
        serializer_objects.insert(0, ExternalImport(import_path=I6_NATIVE_OPS_REPO_PATH))
    return Collection(serializer_objects=serializer_objects, make_local_package_copy=not debug, packages={package})


def serialize_forward(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    forward_module: Optional[str] = None,
    forward_step_name: str = "forward",
    forward_init_args: Optional[Dict[str, Any]] = None,
    unhashed_forward_init_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
    debug: bool = False,
):
    package = PACKAGE + ".pytorch_networks"
    serializer_objects = [
        ExternalImport(import_path=I6_MODELS_REPO_PATH),
        PartialImport(
            code_object_path=package + ".%s.Model" % network_module,
            unhashed_package_root=PACKAGE,
            hashed_arguments=net_args,
            unhashed_arguments=unhashed_net_args or {},
            import_as="get_model",
        ),
    ]
    if include_native_ops:
        serializer_objects.insert(0, ExternalImport(import_path=I6_NATIVE_OPS_REPO_PATH))

    forward_module = forward_module or network_module
    serializer_objects.extend(
        [
            Import(
                code_object_path=package + ".%s.%s_step" % (forward_module, forward_step_name),
                unhashed_package_root=PACKAGE,
                import_as="forward_step",
            ),
            PartialImport(
                code_object_path=package + ".%s.%s_init_hook" % (forward_module, forward_step_name),
                unhashed_package_root=PACKAGE,
                hashed_arguments=forward_init_args or {},
                unhashed_arguments=unhashed_forward_init_args or {},
                import_as="forward_init_hook",
            ),
            Import(
                code_object_path=package + ".%s.%s_finish_hook" % (forward_module, forward_step_name),
                unhashed_package_root=PACKAGE,
                import_as="forward_finish_hook",
            ),
        ]
    )
    return Collection(serializer_objects=serializer_objects, make_local_package_copy=not debug, packages={package})
