from typing import Dict, Tuple


def chunking_with_nfactor(
    chunk_str: str, factor: int, data_key: str = "data", class_key: str = "classes"
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    It gives back the chunking dictionary for different factors. Factor 1 means no subsampling is done
    """

    chunk, overlap = [int(p.strip()) for p in chunk_str.strip().split(":")]
    return ({class_key: chunk // factor, data_key: chunk}, {class_key: overlap // factor, data_key: overlap})
