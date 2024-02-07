from sisyphus import tk
from typing import Any, Dict, Optional

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
)
from i6_experiments.common.setups.serialization import ExternalImport, Import, PartialImport

from .import PACKAGE


def get_pytorch_serializer_v3(
        network_module: str,
        net_args: Dict[str, Any],
        decoder: Optional[str] = None,
        decoder_args: Optional[Dict[str, Any]] = None,
        post_decoder_args: Optional[Dict[str, Any]] = None,
        prior: bool = False,
        debug: bool = False,
        use_custom_engine: bool = False,
        **kwargs
) -> TorchCollection:
    """

    :param network_module: path to the pytorch config file containing Model
    :param net_args: extra arguments for the model
    :param decoder: path to the search decoder, if provided will link search functions
    :param decoder_args: additional arguments to pass to forward_init
    :param post_decoder_args: additional non-hashed arguments to pass to forward_init
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

    # TODO: add flag to switch and maybe move to default tools
    i6_models_repo = CloneGitRepositoryJob(
        url="https://github.com/rwth-i6/i6_models",
        commit="933c6c13f7d6c74e5a59af0257e17c208dae9da3",
        checkout_folder_name="i6_models"
    ).out_repository
    #i6_models_repo = tk.Path("/u/rossenbach/experiments/tts_asr_2023_pycharm/i6_models")
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
    if use_custom_engine:
        pytorch_engine = Import(
            code_object_path=package + ".%s.CustomEngine" % use_custom_engine,
            unhashed_package_root=PACKAGE
        )
        serializer_objects.append(pytorch_engine)
    serializer = TorchCollection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )

    return serializer