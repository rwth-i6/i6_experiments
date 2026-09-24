"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/supervised_reverse_init.py
(``FIT``, ``INIT_HOURS``, ``batch_inventory``, ``batch_loss``, the loop of ``train_step``) and the run
loop of ``sae/emc/blankfree_reverse_init_jobs.BlankfreeSupervisedReverseInitJob``.

ANALYSIS ONLY (uses transcripts): the supervised fit of the reverse model phi on the 10 h seed's gold
phone strings.  It never initialises a main-line arm.

RETURNN side of the supervised reverse init (``supervised.py`` builds the config).  The source ran
its own GPU torch loop; here the SAME fit is a RETURNN training:

* ``get_model`` is :class:`~..model.reverse.SegmentalReverseModel` ``(ReverseConfig())`` alone,
  re-initialised by ``reset_parameters(seed=FIT.seed)`` (the source's init: its own CPU generator,
  so the values do not depend on the device or on the global RNG);
* ``train_step`` computes the source's ``batch_loss`` (``-logp.sum() / frames``, ``frames`` the
  batch's unit frames) through the same function (:func:`_packed_batch_loss`) and marks it as the
  one optimised loss, unnormalised and unscaled, so the gradient RETURNN back-propagates is that
  tensor's.  RETURNN's torch engine then does what the source's ``train_step`` did: backward,
  ``clip_grad_norm_(model.parameters(), 5.0)``, ``torch.optim.Adam(lr=3e-3, weight_decay=0)`` step;
* the batches and their order are the source's ``batch_inventory`` (length buckets of 8, one
  ``torch.randperm`` per epoch from ``torch.Generator().manual_seed(42)``), written by
  ``supervised_data.SupervisedReverseDataJob`` into the HDF in update order and cut back into the
  same batches by :class:`SeedBatchIterDataPipe` (``torch_batching``);
* the held-out NLL per frame (the source's ``held_nll_per_frame``, sum of ``-log p`` over the 28
  held-out utterances / their unit frames) is the error ``nll_per_frame`` on the ``dev`` set,
  written by RETURNN to the ``learning_rates`` file as ``dev_loss_nll_per_frame`` (errors too go
  under ``{dataset}_loss_{name}``); the same key on ``train`` (``train_loss_nll_per_frame``) is the
  source's ``train_nll_per_frame``.  Read ``dev_loss_nll_per_frame`` at epoch 8: ``dev_loss_nll`` is
  a different number (a mean of per-batch values).

The source's grad check (every parameter has a finite gradient) is a post-accumulate-grad hook on
every parameter (:func:`get_model`); the "grad is None" half is not checkable from a hook and every
parameter enters every likelihood.  The source's cost gate (timed profile cases after a warm-up, all
followed by a full reset of weights, RNGs and optimizer) had no effect on the fitted phi and is cut
(``profile_cases``); the 4 h hard budget is the training job's ``time_rqmt``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import torch

from ..model.reverse import (
    NEG_INF,
    FitConfig,
    ReverseConfig,
    SegmentalReverseModel,
    length_buckets,
    assert_finite_sufficient_stats,
    pack_batch,
)

__all__ = [
    "FIT",
    "INIT_HOURS",
    "batch_inventory",
    "batch_loss",
    "get_model",
    "train_step",
    "SeedBatchIterDataPipe",
    "BATCH_ID_KEY",
]

FIT = FitConfig(seed=42)
INIT_HOURS = 4

#: the data key holding each sequence's batch index (``supervised_data`` writes it)
BATCH_ID_KEY = "batch_id"


def batch_inventory(items, cfg=FIT):
    batches = length_buckets(items, cfg.batch_size)
    rng = torch.Generator().manual_seed(cfg.seed)
    schedule = [(epoch, bi) for epoch in range(1, cfg.epochs + 1)
                for bi in torch.randperm(len(batches), generator=rng).tolist()]
    return batches, schedule


def _packed_batch_loss(model, batch, *, full_check):
    """The source's ``batch_loss`` after its ``pack_batch`` line; returns ``(loss, frames, logp)``."""
    logp, stats = model.log_likelihood(**batch, return_stats=True)
    if full_check:
        assert_finite_sufficient_stats(stats)
    elif not bool(torch.isfinite(logp).all()) or bool((logp <= NEG_INF / 2).any()):
        raise FloatingPointError("a seed utterance scored non-finite or impossible")
    frames = int(batch["s_lens"].sum())
    return -logp.sum() / frames, frames, logp


def batch_loss(model, items, *, full_check):
    batch = pack_batch(items, model.cfg, device=next(model.parameters()).device)
    loss, frames, _ = _packed_batch_loss(model, batch, full_check=full_check)
    return loss, frames


# -------------------------------------------------------------------------------------------------
# RETURNN
# -------------------------------------------------------------------------------------------------


def _finite_grad_hook(name: str):
    def hook(param: torch.Tensor) -> None:
        if param.grad is None or not bool(torch.isfinite(param.grad).all()):
            raise FloatingPointError(f"invalid phi gradient: {name}")

    return hook


def get_model(*, epoch: int = None, step: int = None, **_kwargs) -> SegmentalReverseModel:
    """``SegmentalReverseModel(ReverseConfig())`` at the source's init, with the finite-grad check.

    The RETURNN checkpoint of this model is ``{"model": state_dict, ...}`` with the reverse model's
    own (unprefixed) keys, the layout ``reverse_checkpoint_path`` reads.
    """
    model = SegmentalReverseModel(ReverseConfig())
    model.reset_parameters(seed=FIT.seed)
    for name, param in model.named_parameters():
        param.register_post_accumulate_grad_hook(_finite_grad_hook(name))
    return model


#: the train epoch whose first step already ran the full sufficient-statistics check
_FULL_CHECK_DONE: Dict[str, Any] = {"epoch": None}


def train_step(*, model: SegmentalReverseModel, extern_data, **_kwargs):
    """One update of the source's loop (train) or one held-out batch (dev, ``train_flag`` False).

    ``full_check`` as in the source: on the first batch of each train epoch and on every held-out
    batch; the cheap finiteness check otherwise.
    """
    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    ctx = rf.get_run_ctx()
    units, phones, eta = extern_data["units"], extern_data["phones"], extern_data["eta"]
    device = next(model.parameters()).device
    s_dim, u_dim, eta_time = units.dims[1], phones.dims[1], eta.dims[1]
    assert bool((eta_time.dyn_size_ext.raw_tensor == 1).all()), "eta holds one 16-d vector per utterance"
    batch = {
        "z": units.raw_tensor.to(device=device, dtype=torch.long),
        "y": phones.raw_tensor.to(device=device, dtype=torch.long),
        "eta": eta.raw_tensor[:, 0, :].to(device=device, dtype=torch.float32),
        "s_lens": s_dim.dyn_size_ext.raw_tensor.to(device=device, dtype=torch.long),
        "u_lens": u_dim.dyn_size_ext.raw_tensor.to(device=device, dtype=torch.long),
    }
    if ctx.train_flag:
        full_check = _FULL_CHECK_DONE["epoch"] != ctx.epoch
        _FULL_CHECK_DONE["epoch"] = ctx.epoch
    else:
        full_check = True
    loss, _frames, logp = _packed_batch_loss(model, batch, full_check=full_check)
    # the optimised loss: exactly the source's batch_loss tensor (no normalisation, scale 1.0)
    ctx.mark_as_loss(loss, "nll", use_normalized_loss=False)
    # the reported per-frame NLL, pooled over the whole epoch / held-out set:
    # sum over utterances of -log p / sum of their unit frames
    ctx.mark_as_loss(
        -logp.detach(),
        "nll_per_frame",
        dims=[batch_dim],
        as_error=True,
        custom_inv_norm_factor=s_dim.get_size_tensor(),
    )


class SeedBatchIterDataPipe(torch.utils.data.IterDataPipe):
    """``torch_batching``: cut the stream of sequences into the batches ``supervised_data`` wrote.

    Every sequence carries its batch index (:data:`BATCH_ID_KEY`); consecutive sequences with the
    same index form one batch, in the written order.  Train sequence tags are ``e{epoch}/{tag}``,
    and each one is asserted to arrive in its own sub-epoch.
    """

    def __init__(self, dataset: torch.utils.data.IterableDataset, *, train: bool = False, **_fwd_compat):
        super().__init__()
        self._dataset = dataset
        self._train = bool(train)

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        current: List[Dict[str, Any]] = []
        current_id = None
        for data_dict in self._dataset:
            bid = int(data_dict[BATCH_ID_KEY].reshape(-1)[0])
            if self._train:
                tag = str(data_dict["seq_tag"])
                epoch = int(data_dict["epoch"])
                assert tag.startswith(f"e{epoch}/"), f"sequence {tag} arrived in sub-epoch {epoch}"
            if current and bid != current_id:
                yield current
                current = []
            current_id = bid
            current.append(data_dict)
        if current:
            yield current
