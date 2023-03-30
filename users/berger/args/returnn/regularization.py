from typing import Union, Optional, Any, Dict, List


def get_chunking_config(
    base_chunk_size: int = 128,
    chunking_factors: Optional[Union[List[str], Dict[str, int]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    if base_chunk_size <= 0:
        return {}
    if chunking_factors is None:
        return {"chunking": f"{base_chunk_size}:{base_chunk_size//2}"}

    if isinstance(chunking_factors, list):
        chunking_factors = {key: 1 for key in chunking_factors}
    assert isinstance(chunking_factors, Dict)
    return {
        "chunking": (
            {
                key: base_chunk_size // factor
                for key, factor in chunking_factors.items()
            },
            {
                key: base_chunk_size // (2 * factor)
                for key, factor in chunking_factors.items()
            },
        )
    }


def get_base_regularization_config(
    batch_size: int = 10000,
    max_seqs: int = 128,
    accum_grad: int = 1,
    grad_noise: Optional[float] = 0.1,
    grad_clip: Optional[float] = None,
    grad_clip_global_norm: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    result = {"batch_size": batch_size, "max_seqs": max_seqs}
    if grad_noise is not None:
        result["gradient_noise"] = grad_noise
    if grad_clip is not None:
        result["gradient_clip"] = grad_clip
    if grad_clip_global_norm is not None:
        result["gradient_clip_global_norm"] = grad_clip_global_norm
    if accum_grad > 1:
        result["accum_grad_multiple_step"] = accum_grad
    return result
