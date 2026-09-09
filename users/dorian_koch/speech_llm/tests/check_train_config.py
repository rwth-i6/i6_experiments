"""Guard: a typed training config must lower to EXACTLY the hparams dict it replaced.

A Sisyphus job's hash covers its ``hparams`` dict, so a config object that emits one key more or
fewer than the hand-written dict re-hashes the job and re-trains it -- hours of GPU, silently, as a
side effect of a refactor. The 13 existing runs set wildly different key subsets (``rqmt_time_h`` in
13, ``lr`` in 12, ``grad_accum`` in 7, the loss weights in 5, ``warmup_steps`` in 4), so "emit the
field if it was set, and otherwise not at all" is the whole safety property.

Hence every assertion here compares the **exact dict**, never a subset: a subset check would pass
while an extra key silently re-hashed every run.

Also guards that an architecture's default loss does not leak into a run pinned to an older policy
epoch -- the mechanism that lets new runs have real defaults while old ones keep their exact shape.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_train_config.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from speech_llm.full_duplex.sis_recipe.doriank.train_config import (  # noqa: E402
    MOSHI_LIB,
    MOSHI_WEIGHTS,
    PERSONAPLEX_WEIGHTS,
    Compute,
    Loss,
    Optim,
    Probe,
    TextStream,
    _HParams,
    merge_hparams,
)


def check_unset_fields_are_absent():
    """An unset field must not appear AT ALL -- not as None, not as a default."""
    assert Optim().to_hparams() == {}, Optim().to_hparams()
    assert Loss().to_hparams() == {}
    assert Compute().to_hparams() == {}
    assert Probe().to_hparams() == {}
    # One field set -> exactly one key.
    assert Optim(lr=1e-6).to_hparams() == {"lr": 1e-6}
    assert Compute(hours=12).to_hparams() == {"rqmt_time_h": 12}
    # A falsy-but-set value must still be emitted; only None means unset.
    assert Compute(gpus=0).to_hparams() == {"gpu": 0}, "0 is a value, not 'unset'"
    print("PASS  unset fields emit nothing; falsy-but-set values still emit")


def check_probe_data_is_not_an_hparam():
    """Probe.data is a SpeechFinetune constructor argument (so Sisyphus makes the dep edge)."""
    p = Probe(data="/some/probe.jsonl", every=30)
    assert p.to_hparams() == {"knowledge_probe_every": 30}, p.to_hparams()
    print("PASS  Probe.data stays out of hparams (it is a constructor arg)")


def check_real_runs_lower_exactly():
    """The exact dicts three real runs used before the refactor. Byte-for-byte or they re-train."""
    a8_fast = merge_hparams(
        Optim(lr=1e-6, temporal_lr=1e-6, depth_lr=2e-6, grad_accum=4, warmup=500),
        PERSONAPLEX_WEIGHTS,
        Compute(hours=12),
    )
    assert a8_fast == {
        "lr": 1e-6,
        "temporal_lr": 1e-6,
        "depth_lr": 2e-6,
        "audio_other_weight": 0.02,
        "text_pad_weight": 0.3,
        "grad_accum": 4,
        "warmup_steps": 500,
        "rqmt_time_h": 12,
    }, a8_fast

    # a2_lowlr: only two keys. A config object with real defaults would have emitted eight.
    a2 = merge_hparams(Optim(lr=1e-6), Compute(hours=12))
    assert a2 == {"lr": 1e-6, "rqmt_time_h": 12}, a2

    # v3_r8: ONE key -- it sets no lr at all.
    v3_r8 = merge_hparams(Compute(hours=12))
    assert v3_r8 == {"rqmt_time_h": 12}, v3_r8

    # a8_4gpu adds the un-homed marker through extra_hparams.
    g = merge_hparams(Compute(gpus=4, hours=18), extra={"data_pipeline_version": 2})
    assert g == {"gpu": 4, "rqmt_time_h": 18, "data_pipeline_version": 2}, g
    print("PASS  a8_fast / a2_lowlr / v3_r8 / a8_4gpu lower to their exact pre-refactor dicts")


def check_presets_match_the_launcher_defaults_they_mirror():
    """The presets exist to NAME numbers that were copied by hand; they must be the right numbers."""
    assert MOSHI_WEIGHTS.to_hparams() == {"audio_other_weight": 0.01, "text_pad_weight": 0.5}
    assert PERSONAPLEX_WEIGHTS.to_hparams() == {"audio_other_weight": 0.02, "text_pad_weight": 0.3}
    assert MOSHI_LIB.default_loss == MOSHI_WEIGHTS, "moshi's architecture default is its own preset"
    print("PASS  MOSHI_WEIGHTS / PERSONAPLEX_WEIGHTS carry the values they claim to")


def check_arch_default_does_not_leak_into_old_runs():
    """A run pinned to an older epoch must emit NO loss keys when it passes no loss."""
    from speech_llm.full_duplex.sis_recipe.doriank.runs import (
        POLICY_2026_07_31,
        RULE_ARCH_DEFAULTS,
        _policy_applies,
    )

    assert not _policy_applies(POLICY_2026_07_31, RULE_ARCH_DEFAULTS), (
        "an old-epoch run must NOT inherit architecture defaults -- it never set those keys, so "
        "adding them changes its hparams dict and re-trains it"
    )
    # ...and a new run does inherit them.
    from speech_llm.full_duplex.sis_recipe.doriank.runs import POLICY_LATEST

    assert _policy_applies(POLICY_LATEST, RULE_ARCH_DEFAULTS)
    print("PASS  architecture defaults reach new runs only, never older pinned ones")


def check_merge_order():
    """Later parts win, so a run can override its architecture preset."""
    merged = merge_hparams(MOSHI_WEIGHTS, PERSONAPLEX_WEIGHTS)
    assert merged == {"audio_other_weight": 0.02, "text_pad_weight": 0.3}, merged
    print("PASS  later config objects override earlier ones")


def check_every_declared_key_is_rendered():
    """A knob a config object can emit must actually reach a rendered config template.

    ``check_launcher_config_reads.py`` guards the second half of the chain -- every key a template
    RENDERS is read by its launcher -- and nothing guarded the first half, so a knob could be
    declared, hashed, set by a run, and then silently dropped on the floor.

    That is not hypothetical. On 2026-09-09 ``TextStream(emit_epad=...)`` lowered correctly into
    ``hparams`` and re-hashed three arms, but ``finetune.py``'s template renders a FIXED key list and
    had no line for it. All three A21 arms were queued with a config identical to their control's:
    they would have run for 9 GPU-h and produced a null result that meant nothing, and the null would
    have looked like a real answer to "does EPAD matter?".
    """
    import re

    src = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/finetune.py").read_text()
    # Keys as they appear in a rendered YAML template: `<key>: {...}` at the start of a line.
    rendered = set(re.findall(r"^([a-z_][a-z0-9_]*):", src, re.MULTILINE))

    declared: dict[str, str] = {}
    for cls in _HParams.__subclasses__():
        for field_name, key in cls._KEYS.items():
            declared[key] = f"{cls.__name__}.{field_name}"

    # Two declared keys are deliberately NOT config-file keys: Compute lowers to the job's SLURM
    # `rqmt`, which is read by the Sisyphus engine, never written into config.yaml. Exempted by name
    # with the reason, and the exemption is itself checked below so it cannot quietly grow.
    RQMT_KEYS = {"gpu", "rqmt_time_h"}
    for key in RQMT_KEYS:
        assert key in declared, f"{key!r} is exempted as an rqmt key but nothing declares it any more"
    assert not (RQMT_KEYS & rendered), (
        f"{RQMT_KEYS & rendered} is rendered into a config template after all -- drop the exemption "
        f"rather than leaving a real key unchecked"
    )

    missing = {k: v for k, v in declared.items() if k not in rendered and k not in RQMT_KEYS}
    assert not missing, (
        f"declared but never rendered into a config: {missing}. A run can set these, they change "
        f"the job's hash, and the launcher never sees them -- so the arm trains as a copy of its "
        f"control while claiming to test something."
    )
    print(f"PASS  all {len(declared)} declared hparam keys reach a rendered config template")


if __name__ == "__main__":
    check_unset_fields_are_absent()
    check_probe_data_is_not_an_hparam()
    check_real_runs_lower_exactly()
    check_presets_match_the_launcher_defaults_they_mirror()
    check_arch_default_does_not_leak_into_old_runs()
    check_merge_order()
    check_every_declared_key_is_rendered()
    print("ALL PASS")
