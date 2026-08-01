"""Guard: every offline driver still accepts exactly the CLI the harness sends it.

The drivers are invoked as subprocesses (``python -m <offline_module> --in_dir ... --overlay ...``),
so a mismatch between what ``inference_harness.run_offline_driver`` emits and what a driver's parser
declares is not a type error or an import error -- it is an argparse exit(2) on a compute node, after
the job has already queued and allocated a GPU. Nothing upstream can catch it.

That risk went up when the five drivers stopped each declaring their own arguments and started sharing
``moshi_family.offline_cli.build_parser``, because now one edit moves all of them at once. So pin both
directions: every flag the harness can send is accepted, and the defaults that encode a real
per-backend decision (capture window, batch size) still differ where they should.

Checked against the *real* parsers -- each driver's ``main`` is not called; only its parser is built,
which is why this needs no GPU and no model.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_offline_driver_cli.py
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

from moshi_family.offline_cli import build_parser  # noqa: E402

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
HARNESS = SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/inference_harness.py"

#: Flags run_offline_driver can put on the command line, read out of its source so the two cannot
#: drift apart silently. --lora_weights/--lora_config go only to the *fork* drivers (module is None),
#: which are not built on offline_cli, so they are excluded here.
FORK_ONLY = {"--lora_weights", "--lora_config"}
#: --oracle_dataset is passed only when the BackendSpec sets needs_oracle_dataset, which today is KAME
#: alone (knowledge_benchmark.py builds the kwarg from that flag). Asserted rather than assumed: if a
#: second backend ever opts in, this stops matching and the check must be widened instead of silently
#: letting that backend's driver reject a flag it now receives.
BACKENDS = SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/speech_backends.py"
assert BACKENDS.read_text().count("needs_oracle_dataset=True") == 1, (
    "more than one backend now sets needs_oracle_dataset -- add it to ORACLE_BACKENDS below"
)
ORACLE_BACKENDS = {"kame"}

harness_flags = set(re.findall(r'cmd \+= \["(--[a-z_]+)"', HARNESS.read_text()))
harness_flags |= set(re.findall(r'"(--[a-z_]+)", str\(\w+\), "(--[a-z_]+)"', HARNESS.read_text())[0] or ())
harness_flags -= FORK_ONLY | {"--oracle_dataset"}
assert "--in_dir" in harness_flags and "--overlay" in harness_flags, sorted(harness_flags)

#: Flags each backend adds via BackendSpec.offline_extra_args, and the driver each one goes to.
EXTRA_ARGS = {
    "moshi": {"--lora_rank", "--lora_scaling"},
    "moshirag": {"--lora_rank", "--lora_scaling"},
    "kame": {"--hf_repo", "--inject_at_s"},
    "audex": set(),
    "personaplex": set(),
}

#: (module, the per-backend defaults that must stay distinct).
DRIVERS = {
    "moshi": ("moshi_family.offline_inference", {"capture_s": 60.0, "batch_size": 16}),
    "audex": ("moshi_family.audex.offline_inference", {"capture_s": 60.0, "batch_size": 16}),
    "personaplex": ("moshi_family.personaplex.offline_inference", {"capture_s": 60.0, "batch_size": 1}),
    "moshirag": ("moshi_family.moshirag.offline_inference", {"capture_s": 24.0, "batch_size": 1}),
    "kame": ("moshi_family.kame_offline_inference", {"capture_s": 24.0, "batch_size": 1}),
}

#: KAME has no FDB mode (each clip needs an oracle row), so it alone must not take --manifest.
NO_MANIFEST = {"kame"}


def parser_for(module_name: str):
    """Build the driver's parser without running it, by intercepting parse_args."""
    import importlib

    mod = importlib.import_module(module_name)
    captured = {}
    original = type(build_parser("x", default_repo="y")).parse_args

    def intercept(self, *a, **kw):
        captured["parser"] = self
        raise SystemExit(0)

    type(build_parser("x", default_repo="y")).parse_args = intercept
    try:
        mod.main()
    except SystemExit:
        pass
    finally:
        type(build_parser("x", default_repo="y")).parse_args = original
    assert "parser" in captured, f"{module_name}.main() never built a parser"
    return captured["parser"]


failures = []
for tag, (module, expected_defaults) in sorted(DRIVERS.items()):
    p = parser_for(module)
    accepted = {opt for action in p._actions for opt in action.option_strings}
    defaults = {a.dest: a.default for a in p._actions}

    required = harness_flags | EXTRA_ARGS[tag]
    if tag in ORACLE_BACKENDS:
        required |= {"--oracle_dataset"}
    if tag in NO_MANIFEST:
        required -= {"--manifest"}
    missing = sorted(required - accepted)
    if missing:
        failures.append(f"{tag}: harness can send {missing} but the parser rejects them (argparse exit 2 on the node)")

    if tag in NO_MANIFEST and "--manifest" in accepted:
        failures.append(f"{tag}: accepts --manifest but has no FDB path -- it would silently ignore the pairs")
    if tag not in NO_MANIFEST and "--manifest" not in accepted:
        failures.append(f"{tag}: lost --manifest, so every FDB benchmark on this backend dies at startup")

    for key, want in expected_defaults.items():
        got = defaults.get(key)
        if got != want:
            failures.append(
                f"{tag}: default {key}={got!r}, expected {want!r} -- a shared parser flattened a real per-backend choice"
            )

    print(
        f"[ok] {tag:<12} {len(accepted):>2} flags, capture_s={defaults['capture_s']}, batch_size={defaults['batch_size']}"
    )

if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print(f"\nall {len(DRIVERS)} offline drivers accept the harness CLI, with their per-backend defaults intact")
