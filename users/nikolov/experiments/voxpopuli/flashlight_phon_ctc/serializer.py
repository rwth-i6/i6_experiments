import copy
from sisyphus import tk
from typing import Any, Dict, Optional

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
)
from i6_experiments.common.setups.serialization import ExternalImport

#from ..import PACKAGE

from i6_experiments.common.setups.serialization import Import, PartialImport
from ..ctc_rnnt_standalone_2024.default_tools import I6_MODELS_REPO_PATH

PACKAGE = "i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024"


def get_pytorch_serializer_v3(
        network_module: str,
        net_args: Dict[str, Any],
        decoder: Optional[str] = None,
        decoder_args: Optional[Dict[str, Any]] = None,
        post_decoder_args: Optional[Dict[str, Any]] = None,
        prior=False,
        debug=False,
        **kwargs
) -> TorchCollection:
    """

    :param network_module: path to the pytorch config file containing Model
    :param net_args: extra arguments for the model
    :param decoder: path to the search decoder, if provided will link search functions
    :param decoder_args:
    :param post_decoder_args:
    :param prior: build config for prior computation
    :param debug: run training in debug mode (linking from recipe instead of copy)
    :param kwargs:
    :return:
    """
    package = PACKAGE + ".pytorch_networks"

    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments={},
        import_as="get_model",
    )
    pytorch_train_step = Import(
        code_object_path=package + ".%s.train_step" % network_module,
        unhashed_package_root=PACKAGE
    )
    # i6_models_repo = CloneGitRepositoryJob(
    #     url="https://github.com/rwth-i6/i6_models",
    #     commit="1e94a4d9d1aa48fe3ac7f60de2cd7bd3fea19c3e",
    #     checkout_folder_name="i6_models"
    # ).out_repository
    i6_models_repo = tk.Path("/u/rossenbach/experiments/tts_asr_2023_pycharm/i6_models")
    i6_models_repo.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"
    i6_models = ExternalImport(import_path=i6_models_repo)

    serializer_objects = [
        i6_models,
        pytorch_model_import,
        pytorch_train_step,
    ]
    if decoder:
        # Just a hack to test the phoneme-based recognition
        forward_step = Import(
            code_object_path=package + ".%s.forward_step" % decoder,
            unhashed_package_root=PACKAGE,
        )
        init_hook = PartialImport(
            code_object_path=package + ".%s.forward_init_hook" % decoder,
            unhashed_package_root=PACKAGE,
            hashed_arguments=decoder_args or {},
            unhashed_arguments=post_decoder_args or {},
            )
        finish_hook = Import(
            code_object_path=package + ".%s.forward_finish_hook" % decoder,
            unhashed_package_root=PACKAGE,
        )
        serializer_objects.extend(
            [forward_step, init_hook, finish_hook]
        )
    if prior:
        forward_step = Import(
            code_object_path=package + ".%s.prior_step" % network_module,
            unhashed_package_root=PACKAGE,
            import_as="forward_step",
        )
        init_hook = Import(
            code_object_path=package + ".%s.prior_init_hook" % network_module,
            unhashed_package_root=PACKAGE,
            import_as="forward_init_hook",
            )
        finish_hook = Import(
            code_object_path=package + ".%s.prior_finish_hook" % network_module,
            import_as="forward_finish_hook",
            unhashed_package_root=PACKAGE,
        )
        serializer_objects.extend(
            [forward_step, init_hook, finish_hook]
        )
    serializer = TorchCollection(
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
    forward_init_args: Optional[Dict[str, Any]] = None,
    unhashed_forward_init_args: Optional[Dict[str, Any]] = None,
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

    serializer_objects = [
        i6_models,
        pytorch_model_import,
    ]

    forward_module = forward_module or network_module

    forward_step = Import(
        code_object_path=package + ".%s.%s_step" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        import_as="forward_step",
    )
    init_hook = PartialImport(
        code_object_path=package + ".%s.%s_init_hook" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        hashed_arguments=forward_init_args or {},
        unhashed_arguments=unhashed_forward_init_args or {},
        import_as="forward_init_hook",
    )
    finish_hook = Import(
        code_object_path=package + ".%s.%s_finish_hook" % (forward_module, forward_step_name),
        unhashed_package_root=PACKAGE,
        import_as="forward_finish_hook",
    )
    serializer_objects.extend([forward_step, init_hook, finish_hook])

    serializer = TorchCollection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )
    return serializer
