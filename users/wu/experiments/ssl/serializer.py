"""
Serializers for the RETURNN config epilog (model + train_step / forward import), mirroring the
posterior_hmm convention but without RASR/native-ops. See ``serialize_training`` / ``serialize_forward``.
"""

from typing import Any, Dict, Optional

from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.serialization import ExternalImport, Import, PartialImport

from . import PACKAGE
from .default_tools import I6_MODELS_REPO_PATH, RECIPE_ROOT


def serialize_training(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    train_step_args: Optional[Dict[str, Any]] = None,
    unhashed_train_step_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Collection:
    package = PACKAGE + ".pytorch_networks"
    model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )
    train_step = PartialImport(
        code_object_path=package + ".%s.train_step" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=train_step_args or {},
        unhashed_arguments=unhashed_train_step_args or {},
        import_as="train_step",
    )
    i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)
    return Collection(
        serializer_objects=[i6_models, model_import, train_step],
        make_local_package_copy=not debug,
        packages={package},
    )


def serialize_forward(
    network_module: str,
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    forward_module: Optional[str] = None,
    forward_step_name: str = "forward",
    forward_callback_name: str = "ForwardCallback",
    forward_init_args: Optional[Dict[str, Any]] = None,
    unhashed_forward_init_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Collection:
    package = PACKAGE + ".pytorch_networks"
    model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments=unhashed_net_args or {},
        import_as="get_model",
    )
    i6_models = ExternalImport(import_path=I6_MODELS_REPO_PATH)
    recipe_root = ExternalImport(import_path=RECIPE_ROOT)
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
    return Collection(
        serializer_objects=[i6_models, recipe_root, model_import, forward_step, forward_callback],
        make_local_package_copy=not debug,
        packages={package},
    )
