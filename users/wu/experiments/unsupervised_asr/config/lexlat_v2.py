"""Entry points of the lexlat_v2 line (SAE_4A_lexlat_v2.md).

L2-1, stage 1 (main line, label-free; ``reverse_model.phi_first``): phi-first EM under the NULL
recognizer, phi trained alone on the lattice term under the frozen phone trigram.

* :func:`py` -- the PROBE: the three duration settings ``uniform`` / ``durinit`` / ``durfrz`` x seeds
  1, 2, 4 sub-epochs each, every sub-epoch checkpoint scored on the CV holdout by the label-free
  genmarg reads (held-out tau = 1 marginal and posterior decode).  ``durinit`` / ``durfrz`` are the
  main-line use of the duration prior (``BlankfreeDurationPriorMeanJob`` on the train stream; banked
  mean 4.413787447505655 frames): ``reverse_duration_prior`` with mode ``init`` / ``freeze``.
* :func:`wave` -- the 16 restarts, 4 nulls on the structure-destroyed corpus, 2 phi_c restarts, 2
  exact reruns, and the label-free selection ``GenMargSelectionJob``.  NOT BUILT BY :func:`py`.
  Before it can build, three values must be decided and recorded in the phase file:
    - the duration setting (``durinit`` or ``durfrz``; phi_first.WAVE_DURATION_SETTING, from the
      probe read, A9);
    - the sub-epoch count (a multiple of 4; phi_first.WAVE_NUM_SUBEPOCHS, from the A10 read);
    - phi_c's source checkpoint (the banked one is ``k2lat_20_x60`` at sub-epoch 60, an arm the
      port has no preset for).
  ``build_wave`` refuses while the first two are undecided; this entry point takes all three as
  required arguments and never supplies a default.

L2-0, the competence ladder (ANALYSIS ONLY, uses transcripts; ``reverse_model.ladder``): :func:`ladder`
(NOT BUILT BY :func:`py`) -- phi fitted on corrupted / permuted seed gold, their genmarg reads, and
theta at the cold init trained against each phi (nodes R1 / R2), with dev-other greedy PER at the kept
checkpoints.  The random-init phi rung stays unread (``ladder.ladder_phis`` leaves it None: no
writer for a standalone random-init phi checkpoint is registered; deciding one is open).
"""

from __future__ import annotations

from typing import Any, Dict

from sisyphus import tk

__all__ = ["probe", "wave", "ladder", "py"]

_PREFIX = "sae/4a/lexlat_v2"


def _register_genmarg(base: str, files: Dict[str, tk.Path]) -> None:
    for fname, path in files.items():
        tk.register_output(f"{base}/{fname}", path)


def probe(inputs=None) -> dict:
    """``phi_first.build_probe`` on the bed's data, with its scoring reads registered."""
    from ..inputs import get_inputs
    from ..reverse_model.phi_first import build_probe

    inputs = get_inputs() if inputs is None else inputs
    out = build_probe(inputs.data)
    for setting, jobs in out["probes"].items():
        for seed, job in jobs.items():
            base = f"{_PREFIX}/em/probe/{setting}_s{seed:02d}"
            tk.register_output(f"{base}/learning_rates", job.out_learning_rates)
            for ep, files in out["reads"][setting][seed].items():
                _register_genmarg(f"{base}/ep{ep}", files)
    return out


def wave(*, phi_c_source, duration: str, num_subepochs: int, inputs=None) -> dict:
    """The L2-1 wave and its label-free selection (module doc).  Every argument is required: the
    duration setting and sub-epoch count come from the probe / A10 reads and the phase file, and
    ``phi_c_source`` is the whole-model checkpoint phi_c is sliced from.  Nothing here chooses them."""
    from ..inputs import get_inputs
    from ..reverse_model.phi_first import build_wave

    inputs = get_inputs() if inputs is None else inputs
    out = build_wave(inputs.data, phi_c_source=phi_c_source, duration=duration, num_subepochs=num_subepochs)
    sel = out["selection"]["selection"]
    tk.register_output(f"{_PREFIX}/em/wave/selection.json", sel.out_json)
    tk.register_output(f"{_PREFIX}/em/wave/selection.txt", sel.out_report)
    return out


def ladder(inputs=None, *, phi_c_source=None) -> dict:
    """ANALYSIS ONLY (uses transcripts).  L2-0: fits, competence reads, R1 / R2 arms and their
    dev-other PER.  ``phi_c_source`` (optional) adds phi_c's rung; the port builds none by default."""
    from ..analysis.per import epoch_reads
    from ..inputs import get_graph, get_inputs, get_seed_inputs
    from ..reverse_model.ladder import KEEP_EPOCHS, build_fits, build_rt, competence_reads, ladder_phis
    from ..reverse_model.phi_first import reads_bed
    from .common import READ_SPLIT
    from .supervised_init import gold_phi

    inputs = get_inputs() if inputs is None else inputs
    seed = get_seed_inputs()
    data = inputs.data
    fits = build_fits(seed_inputs=seed.seed_inputs, units_hdfs=data["train_units_hdfs"],
                      eta_npz=data["eta_npz"], phi_c_source=phi_c_source)
    phis = ladder_phis(fits, gold_phi=gold_phi(inputs, seed)["checkpoint"])
    competence = competence_reads(phis, reads=reads_bed(data))
    for key, jobs in competence.items():
        if key == "_not_read":
            continue
        cv = jobs["cv_holdout"]
        base = f"{_PREFIX}/ladder/competence/{key}"
        tk.register_output(f"{base}/genmarg.json", cv["marginal"].out_files["genmarg.json"])
        tk.register_output(f"{base}/gendecode.json", cv["decode"].out_files["gendecode.json"])
    rt = build_rt(data=data, graph=get_graph("inhouse_3gram"), phis=phis)
    stream = inputs.dev_stream(READ_SPLIT)
    per: Dict[str, Any] = {}
    for arm, train in rt.items():
        tk.register_output(f"{_PREFIX}/ladder/{arm}/learning_rates", train.out_learning_rates)
        per[arm] = epoch_reads(train, KEEP_EPOCHS, name=f"lexlat_v2_ladder/{arm}", features=stream["features"],
                               originals=stream["originals"], split=READ_SPLIT, gold=inputs.gold)
        for ep, r in per[arm].items():
            tk.register_output(f"{_PREFIX}/ladder/{arm}/ep{ep}/{READ_SPLIT}/per.json", r["per"].out_per)
    return {"fits": fits, "phis": phis, "competence": competence, "rt": rt, "per": per}


def py() -> dict:
    """The main-line L2-1 probe only (see the module doc for :func:`wave` and :func:`ladder`)."""
    return {"probe": probe()}
