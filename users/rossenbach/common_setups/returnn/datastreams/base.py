from sisyphus import tk
from typing import *

from i6_experiments.common.setups.returnn_common.serialization import DataInitArgs, DimInitArgs


class Datastream:
    """
    Defines a "Datastream" for a RETURNN setup, meaning a single entry in the "extern_data" dictionary
    of the RETURNN config.
    """

    def __init__(self, available_for_inference: bool):
        """

        :param available_for_inference: "default" value for "available_for_inference"
        """
        self.available_for_inference = available_for_inference

    def as_returnn_extern_data_opts(
        self, available_for_inference: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        :param available_for_inference: allows to overwrite "available_for_inference" directly
        """
        opts = {
            "available_for_inference": available_for_inference
            if available_for_inference is not None
            else self.available_for_inference,
        }
        return opts

    def as_nnet_constructor_data(
            self, name: str, available_for_inference: Optional[bool] = None, **kwargs
    ):
        d = self.as_returnn_extern_data_opts(
            available_for_inference=available_for_inference
        )
        time_dim = DimInitArgs(
            name="%s_time" % name,
            dim=None,
        )

        dim = d["dim"]

        if d.get("sparse", False):
            sparse_dim = DimInitArgs(
                name="%s_indices" % name,
                dim=dim,
                is_feature=True
            )
            return DataInitArgs(
                name=name,
                available_for_inference=d["available_for_inference"],
                dim_tags=[time_dim],
                sparse_dim=sparse_dim,
            )
        else:
            feature_dim = DimInitArgs(
                name="%s_feature" % name,
                dim=dim,
                is_feature=True,
            )
            return DataInitArgs(
                name=name,
                available_for_inference=d["available_for_inference"],
                dim_tags=[time_dim, feature_dim],
                sparse_dim=None,
            )


class FeatureDatastream(Datastream):
    """
    Defines a datastream for an arbitrary feature
    """

    def __init__(
            self,
            available_for_inference: bool,
            feature_size: Union[tk.Variable, int],
    ):
        """

        :param bool available_for_inference:
        :Param tk.Variable|int embedding_size:
        """
        super().__init__(available_for_inference)
        self.feature_size = feature_size

    def as_returnn_extern_data_opts(
            self, available_for_inference: Optional[bool] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param available_for_inference:
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(
                available_for_inference=available_for_inference
            ),
            "shape": (None, self.feature_size),
            "dim": self.feature_size,
        }
        d.update(kwargs)
        return d
