from dataclasses import dataclass, replace, asdict

from i6_core.returnn.config import ReturnnConfig
from i6_core.serialization import PartialImport
from .... import ROOT_PACKAGE
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection

@dataclass(frozen=True)
class CosineAnnealingLearningRateConfig:
    """
    LR configuration base dataclass.

    Can contain default values.

    Epoch is needed for the method
    """

    num_warmup_epochs: float
    max_lr: float
    min_lr: float

    num_constant_epochs: int = 0

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        assert self.max_lr >= self.min_lr, "max_lr must be >= min_lr"

    def get_cosine_annealing_lr_config(self, epochs: int, debug:bool, **kwargs) -> ReturnnConfig:
        dyn_lr_import = PartialImport(
            code_object_path="i6_experiments.users.juanola.training.lr_schedules.cosine_annealing.linear_warmup_cosine_annealing",
            import_as="dynamic_learning_rate",
            hashed_arguments=dict(sorted({"num_epochs": epochs, **asdict(self)}.items())),
            unhashed_arguments={},
            unhashed_package_root=None,
        )

        prolog = Collection(
            serializer_objects=[dyn_lr_import],
            make_local_package_copy=not debug,
            packages={ROOT_PACKAGE},
        )

        return ReturnnConfig(
            config={"learning_rate_control": "constant"},
            python_prolog=prolog,
        )


"""
parameter groups
"""

_BASE_CA_LRS_KWARGS = dict(
    num_warmup_epochs=0.1,
    max_lr=1e-5,
    min_lr=1e-7,
)

"""
Specific configurations set below.
"""


def ca_lr_baseline_pk5() -> CosineAnnealingLearningRateConfig:
    return CosineAnnealingLearningRateConfig(**_BASE_CA_LRS_KWARGS)


def ca_lr_baseline_pk4() -> CosineAnnealingLearningRateConfig:
    return replace(ca_lr_baseline_pk5(), max_lr=1e-4)


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
