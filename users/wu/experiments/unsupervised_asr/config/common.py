"""Shared wiring of the ``config/`` entry points: one arm's training job and its lean reads.

Reads per arm (dev-other; ``analysis/``):

* at every kept checkpoint: theta slice -> posterior dump (``ReturnnForwardJobV2``) -> greedy PER
  (``analysis.per.epoch_reads``);
* at the FINAL checkpoint only: the derangement gap and the D4 decode gap under the arm's own
  ``reverse.`` slice (``analysis.gaps``).  The decode gap re-derives the derangement column and is
  checked against the derangement job's ``derangement_gap.json`` of the same arm and epoch
  (``reference_gap``).

The JS rows (``analysis.jsd.JsRowsReadJob``) and the paired PER deltas are built by the entry points
over several arms (:func:`js_rows`, :func:`paired_delta`).

Output names are ``sae/4a/<arm>/...``; registering the same output twice (two entry points loaded in
one manager) is a no-op in sisyphus, and equal jobs are merged by hash.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from sisyphus import tk

__all__ = ["READ_SPLIT", "JS_ROWS_N_BOOT", "JS_ROWS_SEED", "train_and_read", "js_rows", "paired_delta"]

#: the split of every read (the phase's banked PER numbers are dev-other)
READ_SPLIT = "dev-other"
#: ``analysis.jsd.JS_ROWS_CONVENTION``'s registered bootstrap: 2000 speaker resamples at seed 0
JS_ROWS_N_BOOT = 2000
JS_ROWS_SEED = 0

_PREFIX = "sae/4a"


def train_and_read(arm, inputs, *, gaps: bool = True) -> Dict[str, Any]:
    """The ``ReturnnTrainingJob`` of ``arm`` (``training.arms.Arm``) and its reads.

    :param inputs: ``inputs.get_inputs()``.
    :param gaps: build the final-checkpoint derangement / decode gaps.
    :return: ``{"arm", "train", "per": {epoch: {"theta", "post", "per"}}, "final", "phi", "gaps"}``.
    """
    from ..analysis.gaps import decode_gap, derangement_gap
    from ..analysis.per import epoch_reads
    from ..training.checkpoints import ExtractSubmoduleCheckpointJob
    from ..training.jobs import train_arm

    name = arm.name
    train = train_arm(**arm.job_kwargs())
    tk.register_output(f"{_PREFIX}/{name}/learning_rates", train.out_learning_rates)

    stream = inputs.dev_stream(READ_SPLIT)
    reads = epoch_reads(train, arm.keep_epochs, name=name, features=stream["features"],
                        originals=stream["originals"], split=READ_SPLIT, gold=inputs.gold)
    for epoch, r in reads.items():
        base = f"{_PREFIX}/{name}/ep{epoch}/{READ_SPLIT}"
        tk.register_output(f"{base}/per.json", r["per"].out_per)
        tk.register_output(f"{base}/per.txt", r["per"].out_report)

    final = max(arm.keep_epochs)
    assert final == arm.num_epochs, (name, final, arm.num_epochs)
    out: Dict[str, Any] = {"arm": arm, "train": train, "per": reads, "final": final, "gaps": {}}
    if not gaps:
        return out

    ckpt = train.out_checkpoints[final]
    phi = ExtractSubmoduleCheckpointJob(checkpoint=ckpt.path if hasattr(ckpt, "path") else ckpt,
                                        prefix="reverse.")
    phi.add_alias(f"{_PREFIX}/{name}/ep{final}/phi")
    out["phi"] = phi
    per = reads[final]["per"]
    common = dict(reverse_checkpoint=phi.out_checkpoint, raw_hyps=per.out_raw_hyps, gold=inputs.gold,
                  units=stream["units"], eta_npz=inputs.data["eta_npz"], split=READ_SPLIT)
    der = derangement_gap(name=f"{name}/ep{final}/{READ_SPLIT}", **common)
    dec = decode_gap(name=f"{name}/ep{final}/{READ_SPLIT}", reference_gap=der["gap"].out_summary, **common)
    base = f"{_PREFIX}/{name}/ep{final}/{READ_SPLIT}"
    tk.register_output(f"{base}/derangement_gap.json", der["gap"].out_summary)
    tk.register_output(f"{base}/decode_gap.json", dec["gap"].out_summary)
    tk.register_output(f"{base}/decode_gap.txt", dec["gap"].out_report)
    out["gaps"] = {"derangement": der, "decode": dec}
    return out


def _decode(read: Dict[str, Any], epoch: int) -> Dict[str, tk.Path]:
    per = read["per"][epoch]["per"]
    return {"raw": per.out_raw_hyps, "phones": per.out_hyps, "per": per.out_per}


def js_rows(name: str, reads: Dict[str, Dict[str, Any]], inputs,
            rows: Sequence[Dict[str, Any]] = ()) -> Any:
    """``JsRowsReadJob`` over the FINAL-checkpoint decodes of ``reads`` (``{arm: train_and_read(...)}``)
    and the ``cand - base`` ``rows`` (``{"tag", "cand", "base"}``, each at its arm's final epoch)."""
    from ..analysis.jsd import JsRowsReadJob
    from ..analysis.paired import EXPECTED_UTTS

    decodes = {arm: {r["final"]: _decode(r, r["final"])} for arm, r in reads.items()}
    full_rows = [dict(tag=row["tag"], cand=row["cand"], cand_epoch=reads[row["cand"]]["final"],
                      base=row["base"], base_epoch=reads[row["base"]]["final"]) for row in rows]
    job = JsRowsReadJob(decodes=decodes, rows=full_rows, gold=inputs.gold, split=READ_SPLIT,
                        expected_utterances=EXPECTED_UTTS[READ_SPLIT], prior_npz=inputs.prior_npz,
                        n_boot=JS_ROWS_N_BOOT, seed=JS_ROWS_SEED)
    job.add_alias(f"{_PREFIX}/js_rows/{name}")
    tk.register_output(f"{_PREFIX}/js_rows/{name}/js_rows.json", job.out_json)
    tk.register_output(f"{_PREFIX}/js_rows/{name}/report.txt", job.out_report)
    return job


def paired_delta(cand: Dict[str, Any], base: Dict[str, Any], inputs, *, epoch: Optional[int] = None) -> Any:
    """``PairedPerDeltaJob`` of ``cand`` against the BASELINE ``base`` (``train_and_read`` outputs) at
    ``epoch`` (default: the candidate's final epoch, which must be kept by both).  Convention of the
    job: ``per_a`` = baseline, ``per_b`` = candidate, a negative delta favours the candidate."""
    from ..analysis.paired import PairedPerDeltaJob

    epoch = cand["final"] if epoch is None else int(epoch)
    c, b = cand["arm"].name, base["arm"].name
    assert epoch in cand["per"] and epoch in base["per"], (c, b, epoch)
    tag = f"{c}_vs_{b}"
    job = PairedPerDeltaJob(per_a=base["per"][epoch]["per"].out_hyps, per_b=cand["per"][epoch]["per"].out_hyps,
                            gold=inputs.gold, split=READ_SPLIT, name=f"{tag}/ep{epoch}", baseline_name=b)
    job.add_alias(f"{_PREFIX}/paired/{tag}/ep{epoch}/{READ_SPLIT}")
    base_out = f"{_PREFIX}/paired/{tag}/ep{epoch}/{READ_SPLIT}"
    tk.register_output(f"{base_out}/paired_per.json", job.out_paired_per)
    tk.register_output(f"{base_out}/summary.txt", job.out_summary)
    return job
