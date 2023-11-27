__all__ = ["get_shared_time_tag", "get_context_dim_tag_prolog"]


from textwrap import dedent
from typing import Tuple

from i6_core import returnn


def get_default_time_tag_str() -> str:
    return "__time_tag__"

def get_shared_time_tag() -> Tuple[str, str]:
    var_name = get_default_time_tag_str()
    code = dedent(
        f"""
        from returnn.tf.util.data import Dim
        {var_name} = Dim(dimension=None, kind=Dim.Types.Spatial, description="time")
        """
    )
    return code, var_name


def get_context_dim_tag_prolog(
    spatial_size: int,
    feature_size: int,
    context_type: str,
    spatial_dim_variable_name: str,
    feature_dim_variable_name: str,
) -> Tuple[str, returnn.CodeWrapper, returnn.CodeWrapper]:
    code = dedent(
        f"""
        from returnn.tf.util.data import FeatureDim, SpatialDim
        {spatial_dim_variable_name} = SpatialDim("contexts-{context_type}", {spatial_size})
        {feature_dim_variable_name} = FeatureDim("{context_type}", {feature_size})
        """
    )
    return (
        code,
        returnn.CodeWrapper(spatial_dim_variable_name),
        returnn.CodeWrapper(feature_dim_variable_name),
    )
