"""Every LoRA overlay reaching the offline driver must come with its rank.

THE BUG THIS EXISTS FOR (2026-09-14, cost 40 errored jobs + a stalled graph):
``benchmarks.py`` attached the n=1000 judged eval to every ft_a21 arm with
``moshi_family_backend_spec()`` and no ``lora_rank``. The driver wraps the linears with
``LoraConfig`` BEFORE loading ``--overlay``, so with no rank it builds a plain model and all 674 of
the adapter's keys are "absent from the model" -- an ``AssertionError`` raised ~25 s in, AFTER the
GPU has been allocated. Sisyphus then refuses to run a config containing errored jobs, so 40
failures held back everything else in the graph.

Why no existing guard caught it: ``check_overlay_resolution.py`` guards the CHECKPOINT LAYOUT
(``overlay_kind`` lora-vs-full, so the resolver symlinks a file the run actually wrote). That is the
other half of the same wiring and it was green throughout -- the layout was right, the rank was
missing. The two are set at different call sites and only this one is checked against the graph.

This walks the REAL graph rather than a fixture, because the defect was a single call site that
looked exactly like its eleven correct neighbours; a fixture would have been written from the
correct ones. Cost is one config load (~1 min), so run it after touching any ``*_backend_spec``
call or adding an arm, not on every edit.
"""

import os
import sys
import logging

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import tk
from sisyphus.loader import config_manager

CONFIG = "recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"


def _finished(job) -> bool:
    p = job._sis_path()
    return os.path.exists(p + "/finished") or os.path.exists(p + "/finished.tar.gz")


def main() -> int:
    config_manager.load_configs([CONFIG])

    checked = 0
    broken = []
    for job in tk.sis_graph.jobs():
        if type(job).__name__ != "SpeechInference":
            continue
        # A finished job already proved itself; re-flagging it would make the guard un-greenable
        # for history we cannot change.
        if _finished(job):
            continue
        if getattr(job, "lora_weights", None) is None:
            continue  # base model / no overlay -- nothing to wrap, correctly no rank
        checked += 1
        extra = tuple(getattr(job, "offline_extra_args", ()) or ())
        if "--lora_rank" not in extra:
            broken.append(job._sis_id().split(".")[-1])

    if broken:
        print(f"FAIL  {len(broken)} of {checked} pending overlay jobs carry no --lora_rank:")
        for h in broken[:10]:
            print(f"        SpeechInference.{h}")
        print()
        print("      The driver builds LoRALinears from --lora_rank BEFORE loading --overlay, so")
        print("      these will die on 'overlay has N key(s) absent from the model' after the GPU")
        print("      is allocated. Pass the rank from the run, as attach_evals does:")
        print("          moshi_family_backend_spec(")
        print("              lora_rank=None if run.overlay_kind == 'full' else run.lora_rank)")
        return 1

    print(f"PASS  all {checked} pending SpeechInference jobs with a LoRA overlay carry --lora_rank")

    # Non-vacuous: the check must be able to FAIL. Build the spec the broken call site built and
    # assert it really produces no rank -- otherwise a change making every spec carry a rank
    # unconditionally would leave this guard green for the wrong reason.
    from i6_experiments.users.dorian_koch.speech_llm.speech_backends import (
        moshi_family_backend_spec,
    )

    assert "--lora_rank" not in tuple(moshi_family_backend_spec().offline_extra_args), (
        "moshi_family_backend_spec() with no rank should emit no --lora_rank; if it now defaults "
        "to one, this guard can no longer fail and must be rewritten."
    )
    assert "--lora_rank" in tuple(moshi_family_backend_spec(lora_rank=128).offline_extra_args)
    print("PASS  the guard can still fail (rank-less spec really emits no --lora_rank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
