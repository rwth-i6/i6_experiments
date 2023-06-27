import string
import textwrap
from typing import Any, Dict
from i6_core.util import instanciate_delayed
from i6_experiments.common.setups.serialization import SerializerObject
from sisyphus.hash import sis_hash_helper


class ExportPyTorchModel(SerializerObject):
    """
    Serializes a `get_network` function into the config, which calls
    a defined network construction function and defines the parameters to it.
    This is for returnn_common networks.

    Note that the network constructor function always needs "epoch" as first defined parameter,
    and should return an `nn.Module` object.
    """

    TEMPLATE = textwrap.dedent(
        """\

    model_kwargs = ${MODEL_KWARGS}

    def get_model():
        return ${MODEL_CLASS}(epoch=${EPOCH}, step=${STEP}, **${MODEL_KWARGS})

    """
    )

    def __init__(
        self,
        model_class_name: str,
        epoch: int,
        step: int,
        model_kwargs: Dict[str, Any],
    ):
        """
        :param model_class_name:
        :param model_kwargs:
        """

        super().__init__()
        self.model_class_name = model_class_name
        self.epoch = epoch
        self.step = step
        self.model_kwargs = model_kwargs

    def get(self):
        """get"""
        return string.Template(self.TEMPLATE).substitute(
            {
                "MODEL_KWARGS": str(instanciate_delayed(self.model_kwargs)),
                "EPOCH": self.epoch,
                "STEP": self.step,
                "MODEL_CLASS": self.model_class_name,
            }
        )

    def _sis_hash(self):
        h = {
            "epoch": self.epoch,
            "step": self.step,
            "model_kwargs": self.model_kwargs,
        }
        return sis_hash_helper(h)
