__all__ = ["subsample_chunking"]

import typing


def subsample_chunking(
    ch: str, factor: int, data_key: str = "data", subsampled_key: str = "classes"
) -> typing.Tuple[typing.Dict[str, int], typing.Dict[str, int]]:
    parts = [int(p.strip()) for p in ch.strip().split(":")]
    size, step = parts

    # ceil div
    size_part = {data_key: size, subsampled_key: (size + factor - 1) // factor}
    step_part = {data_key: step, subsampled_key: (step + factor - 1) // factor}

    return size_part, step_part
