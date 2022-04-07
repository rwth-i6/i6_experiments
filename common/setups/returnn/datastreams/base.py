from typing import *
from sisyphus import tk


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

    def as_returnn_common_data_and_dims(
        self, name: str, available_for_inference: Optional[bool] = None, **kwargs
    ):
        """
        :param datastream name (e.g. the datastream key)
        :param available_for_inference:
        :param kwargs:
        """
        d = self.as_returnn_extern_data_opts(
            available_for_inference=available_for_inference
        )
        from returnn_common import nn

        time_dim = nn.SpatialDim("%s_time" % name)
        if isinstance(d["dim"], tk.Variable):
            # for dynamic dims we need asynchron workflows
            tk.async_run(d["dim"])
            assert False
            dim = d["dim"].get()
        else:
            assert isinstance(d["dim"], int)
            dim = d["dim"]
        if d.get("sparse", False):
            sparse_dim = nn.FeatureDim("%s_indices", dimension=dim)
            data = nn.Data(
                name=name,
                available_for_inference=d["available_for_inference"],
                dim_tags=[nn.batch_dim, time_dim],
                sparse_dim=sparse_dim,
                **kwargs,
            )
            return data, [time_dim, sparse_dim]
        else:
            feature_dim = nn.FeatureDim("%s_feature", dimension=dim)
            data = nn.Data(
                name=name,
                available_for_inference=d["available_for_inference"],
                dim_tags=[nn.batch_dim, time_dim, feature_dim],
                **kwargs,
            )
            return data, [time_dim, feature_dim]
