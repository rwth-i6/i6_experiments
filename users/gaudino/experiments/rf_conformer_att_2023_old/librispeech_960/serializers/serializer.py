import copy
from sisyphus import tk
from typing import Any, Dict, Optional, List

from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    ExternData,
    Import,
    Network,
    PythonEnlargeStackWorkaroundNonhashedCode,
    ExplicitHash
)

from .basic import (
    get_basic_pt_network_serializer,
    get_basic_pt_network_train_serializer,
    get_basic_pt_network_recog_serializer
)
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
    PyTorchModel,
)

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.serializer import Import as ImportV2, PartialImport
from ..models.torchaudio_conformer_ctc import ConformerCTCConfig

PACKAGE = __package__

def get_pytorch_serializer(
        network_module: str,
        net_args: Dict[str, Any],
        forward=False,
        debug=False,
        **kwargs
) -> TorchCollection:

    package = PACKAGE

    pytorch_model_import = Import(
        package + ".%s.Model" % network_module
    )
    pytorch_train_step = Import(
        package + ".%s.train_step" % network_module
    )
    pytorch_model = PyTorchModel(
        model_class_name=pytorch_model_import.object_name,
        model_kwargs=net_args,
    )
    serializer_objects = [
        pytorch_model_import,
        pytorch_train_step,
        pytorch_model,
    ]
    if forward:
        forward_step = Import(
            package + ".%s.forward_step" % network_module
        )
        serializer_objects.extend([forward_step])

        init_hook = Import(
            package + ".%s.forward_init_hook" % network_module
        )
        finish_hook = Import(
            package + ".%s.forward_finish_hook" % network_module
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

def get_pytorch_serializer_v2(
        network_module: str,
        net_args: Dict[str, Any],
        forward=False,
        search=False,
        debug=False,
        **kwargs
) -> TorchCollection:

    package = PACKAGE

    pytorch_model_import = PartialImport(
        code_object_path=package + ".%s.Model" % network_module,
        unhashed_package_root=PACKAGE,
        hashed_arguments=net_args,
        unhashed_arguments={},
        import_as="get_model",
    )
    pytorch_train_step = ImportV2(
        code_object_path=package + ".%s.train_step" % network_module,
        unhashed_package_root=PACKAGE
    )

    serializer_objects = [
        pytorch_model_import,
        pytorch_train_step,
    ]

    if forward:
        forward_step = Import(
            package + ".%s.forward_step" % network_module
        )
        init_hook = Import(
            package + ".%s.forward_init_hook" % network_module
        )
        finish_hook = Import(
            package + ".%s.forward_finish_hook" % network_module
        )
        serializer_objects.extend(
            [forward_step, init_hook, finish_hook]
        )
    elif search:
        # Just a hack to test the phoneme-based recognition
        forward_step = Import(
            package + ".%s.search_step" % network_module,
            import_as="forward_step"
        )
        init_hook = PartialImport(
            code_object_path=package + ".%s.search_init_hook" % network_module,
            unhashed_package_root=PACKAGE,
            hashed_arguments=kwargs["search_args"],
            unhashed_arguments={},
            import_as="forward_init_hook",
            )
        finish_hook = Import(
            package + ".%s.search_finish_hook" % network_module,
            import_as="forward_finish_hook"
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


# Serializers for the torchaudio.conformer
def get_serializer(
    model_config: ConformerCTCConfig,
    model_import_path: str,
    train: bool = True,
    import_kwargs: Dict[str, Any] = None,
    model_kwargs: Dict[str, Any] = None,
    debug=False,
    additional_train_serializer_objects: Optional[List] = None,
    additional_forward_serializer_objects: Optional[List] = None,
) -> TorchCollection:

    # PACKAGE = i6_private.users.gruev.experiments.pytorch_ctc_2023.librispeech.pytorch_conformer_ctc
    PACKAGE = __package__.rsplit('.', 1)[0]

    if train:
        return get_basic_pt_network_train_serializer(
            module_import_path=f"{PACKAGE}.{model_import_path}.ConformerCTCModel",
            train_step_import_path=f"{PACKAGE}.{model_import_path}.train_step",
            model_config=model_config,
            model_kwargs=model_kwargs,
            debug=debug,
            additional_packages=[PACKAGE, "i6_models"],
            additional_serializer_objects=additional_train_serializer_objects,
        )
    else:
        return get_basic_pt_network_recog_serializer(
            module_import_path=f"{PACKAGE}.{model_import_path}.ConformerCTCModel",
            recog_step_import_path=f"{PACKAGE}.{model_import_path}",
            model_config=model_config,
            import_kwargs=import_kwargs,
            model_kwargs=model_kwargs,
            debug=debug,
            additional_packages=[PACKAGE, "i6_models"],
            additional_serializer_objects=additional_forward_serializer_objects,
        )

# ### Serializers for the i6_models.conformer
# def get_train_serializer(
#     model_config: ConformerCTCConfig,
# ) -> Collection:
#     pytorch_package = __package__.rpartition(".")[0]
#     return get_basic_pt_network_serializer(
#         module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
#         model_config=model_config,
#         additional_serializer_objects=[
#             Import(f"{pytorch_package}.train_steps.ctc.train_step"),
#         ],
#     )
#
#
# def get_prior_serializer(
#     model_config: ConformerCTCConfig,
# ) -> Collection:
#     pytorch_package = __package__.rpartition(".")[0]
#     return get_basic_pt_network_serializer(
#         module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
#         model_config=model_config,
#         additional_serializer_objects=[
#             Import(f"{pytorch_package}.forward.basic.forward_step"),
#             Import(f"{pytorch_package}.forward.prior_callback.ComputePriorCallback", import_as="forward_callback"),
#         ],
#     )
#
#
# def get_recog_serializer(
#     model_config: ConformerCTCConfig,
# ) -> Collection:
#     pytorch_package = __package__.rpartition(".")[0]
#     return get_basic_pt_network_serializer(
#         module_import_path=f"{__name__}.{ConformerCTCModel.__name__}",
#         model_config=model_config,
#         additional_serializer_objects=[
#             Import(f"{pytorch_package}.export.ctc.export"),
#         ],
#     )
#
#
# def get_serializer_v2(model_config: ConformerCTCConfig, variant: ConfigVariant) -> Collection:
#     if variant == ConfigVariant.TRAIN:
#         return get_train_serializer(model_config)
#     if variant == ConfigVariant.PRIOR:
#         return get_prior_serializer(model_config)
#     if variant == ConfigVariant.ALIGN:
#         return get_recog_serializer(model_config)
#     if variant == ConfigVariant.RECOG:
#         return get_recog_serializer(model_config)
#     raise NotImplementedError
