import copy
from sisyphus import tk
from typing import Any, Dict

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
)
from i6_experiments.common.setups.serialization import ExternalImport

PACKAGE = __package__

from i6_experiments.users.rossenbach.common_setups.returnn.serializer import Import, PartialImport


def get_pytorch_serializer_v3(
        network_module: str,
        net_args: Dict[str, Any],
        use_custom_engine=False,
        search=False,
        debug=False,
        search_args: Dict[str, Any]={},
        **kwargs
) -> TorchCollection:

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
    # i6_models_repo = tk.Path("/u/rossenbach/experiments/tts_asr_2023_pycharm/i6_models")
    # i6_models_repo.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"
    # i6_models = ExternalImport(import_path=i6_models_repo)

    serializer_objects = [
        # i6_models,
        pytorch_model_import,
        pytorch_train_step,
    ]
    if search:
        # Just a hack to test the phoneme-based recognition
        forward_step = Import(
            code_object_path=package + ".%s.forward_step" % network_module,
            unhashed_package_root=PACKAGE,
        )
        init_hook = PartialImport(
            code_object_path=package + ".%s.forward_init_hook" % network_module,
            unhashed_package_root=PACKAGE,
            hashed_arguments=search_args or {},
            unhashed_arguments={}
            )
        finish_hook = Import(
            code_object_path=package + ".%s.forward_finish_hook" % network_module,
            unhashed_package_root=PACKAGE,
        )
        serializer_objects.extend(
            [forward_step, init_hook, finish_hook]
        )
    if use_custom_engine:
        pytorch_engine = Import(
            code_object_path=package + ".%s.CustomEngine" % network_module,
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