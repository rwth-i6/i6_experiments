__all__ = ["subsample_chunking"]

import typing


def subsample_chunking(
    ch: str, factor: int, data_key: str = "data", subsampled_key: typing.Union[str, typing.List[str]] = "classes"
) -> typing.Tuple[typing.Dict[str, int], typing.Dict[str, int]]:
    parts = [int(p.strip()) for p in ch.strip().split(":")]
    assert all(
        (p % factor == 0 for p in parts)
    ), "factor must evenly divide chunk size, set chunk size to be next multiple of factor to avoid this issue"
    size, step = parts

    subsampled_key = subsampled_key if isinstance(subsampled_key, list) else [subsampled_key]

    size_part = {k: size // factor for k in subsampled_key}
    size_part = {**size_part, data_key: size}
    step_part = {k: step // factor for k in subsampled_key}
    step_part = {**step_part, data_key: step}

    return size_part, step_part
