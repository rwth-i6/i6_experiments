from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


def alternate_batching(
    dataset: torch.utils.data.IterableDataset, *, train: bool, **_kwargs
) -> torch.utils.data.IterableDataset:
    """
    Alternate batching, alternating between two kinds, ASR (speech+text) and LM (text-only).

    This is usually used together with :class:`CombinedDataset` with ``seq_ordering="interleave"``.

    For :class:`CombinedDataset`, also remember to set ``batch_size`` appropriately,
    i.e. as a dict, defining the batch size for both data keys (speech and text).

    Example usage of :func:`alternate_batching`::

        train_config["torch_batching"] = alternate_batching
        train_config["accum_grad_multiple_step"] *= 2

    """

    from returnn.config import get_global_config
    from returnn.torch.data.pipeline import BatchingIterDataPipe
    from .torch.alternate_batching import AlternateBatchingIterDataPipe

    config = get_global_config()
    batch_size = config.typed_value("batch_size", -1)
    batch_size = config.typed_value(f"batch_size_{'train' if train else 'dev'}", batch_size)
    assert batch_size != -1, f"batch_size or batch_size_{'train' if train else 'dev'} not defined in config"
    max_seqs = config.typed_value("max_seqs", -1)

    if not train:
        return BatchingIterDataPipe(dataset, batch_size=batch_size, max_seqs=max_seqs)

    return AlternateBatchingIterDataPipe(dataset, batch_size=batch_size, max_seqs=max_seqs)
