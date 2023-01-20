__all__ = ["get_shared_time_tag"]


from textwrap import dedent
import typing


def get_shared_time_tag() -> typing.Tuple[str, str]:
    var_name = "__time_tag__"
    code = dedent(
        f"""
        from TFUtil import DimensionTag
        {var_name} = DimensionTag(kind=DimensionTag.Types.Spatial, description="time")
        """
    )
    return code, var_name
