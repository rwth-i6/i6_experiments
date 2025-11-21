from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


def alternate_batching(
    dataset: torch.utils.data.IterableDataset, *, train: bool, **_kwargs
) -> torch.utils.data.IterableDataset:
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
