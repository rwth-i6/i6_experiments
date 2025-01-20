import os
from typing import List, Type

from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    PyTorchModel,
    build_config_constructor_serializers_v2,
)
from i6_experiments.common.setups.serialization import Collection, ExternalImport, Import
from i6_models.config import ModelConfiguration, ModuleType
from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase

recipe_imports = [
    "import sys",
    ExternalImport(
        import_path=tk.Path(
            f"{os.path.realpath(__file__).split('recipe')[0]}recipe/",
            hash_overwrite="RECIPE_ROOT",
        )
    ),
]


def get_model_serializers(model_class: Type[ModuleType], model_config: ModelConfiguration) -> List[DelayedBase]:
    constructor_call, model_imports = build_config_constructor_serializers_v2(
        cfg=model_config,
        variable_name="cfg",
    )

    model_serializers: List[DelayedBase] = [
        Import(
            f"{model_class.__module__}.{model_class.__name__}",
        ),
    ]
    model_serializers.append(Collection(model_imports))  # type: ignore
    model_serializers.append(constructor_call)
    model_serializers.append(
        PyTorchModel(
            model_class_name=model_class.__name__,
            model_kwargs={"cfg": CodeWrapper("cfg")},
        )
    )

    return model_serializers
