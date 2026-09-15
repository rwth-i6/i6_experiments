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
    "flmaudio": {"--hf_repo"},
}

#: (module, the per-backend defaults that must stay distinct).
DRIVERS = {
    "moshi": ("moshi_family.offline_inference", {"capture_s": 60.0, "batch_size": 16}),
    "audex": ("moshi_family.audex.offline_inference", {"capture_s": 60.0, "batch_size": 16}),
    "personaplex": ("moshi_family.personaplex.offline_inference", {"capture_s": 60.0, "batch_size": 1}),
    "moshirag": ("moshi_family.moshirag.offline_inference", {"capture_s": 24.0, "batch_size": 1}),
    "kame": ("moshi_family.kame_offline_inference", {"capture_s": 24.0, "batch_size": 1}),
    # flmaudio is the one LIB driver NOT built on offline_cli -- it is a foreign backbone vendored
    # for inference only and carries its own argparse. That is exactly why it belongs here: the
    # harness decides what to send from `module is not None`, not from which parser the driver uses,
    # so a flag added to offline_cli's shared parser silently skips this one. It did: --seed was
    # emitted for every `offline_module=` backend and this driver did not declare it, which is an
    # argparse exit(2) after the GPU is allocated. Found 2026-09-15 by reading, before it fired.
    "flmaudio": ("flmaudio.offline_inference", {"capture_s": 24.0, "batch_size": 1}),
}

#: KAME has no FDB mode (each clip needs an oracle row), so it alone must not take --manifest.
NO_MANIFEST = {"kame"}

#: Drivers that legitimately never receive --overlay, because no adapter is ever resolved for them.
#: FLM-Audio is an external HF checkpoint we only run inference on (FDB_MODEL_ORIGIN: "hf"), so the
#: harness never has lora_weights to pass -- and --overlay is emitted only `if lora_weights is not
#: None`. Exempting it is safe ONLY while that stays true, so the exemption is checked below rather
#: than trusted.
NO_OVERLAY = {"flmaudio"}
assert "lora" not in BACKENDS.read_text().split("def flm_audio_backend_spec")[1].split("def ")[0], (
    "flm_audio_backend_spec now mentions lora -- it may receive --overlay, so drop it from NO_OVERLAY"
)


#: Drivers that run in an ISOLATED venv and therefore cannot be imported by this check.
#: flmaudio pins its own `transformers` (the setup venv raises `cannot import name 'LossKwargs'`),
#: so its parser is read from SOURCE instead. That is a weaker check than building the real parser
#: -- it cannot see a flag added dynamically -- but it is far stronger than skipping the driver,
#: which is what let the --seed gap exist in the first place. Keyed by module -> source path.
SOURCE_ONLY = {"flmaudio.offline_inference": "flmaudio/offline_inference.py"}


def _parser_from_source(rel_path: str):
    """(accepted option strings, {dest: default}) read out of the driver's `add_argument` calls.

    Used only for SOURCE_ONLY drivers. Returns the same two things the import path derives from a
    real parser, so the assertions below are identical either way.
    """
    import ast

    src = (SETUP / "recipe/speech_llm/full_duplex" / rel_path).read_text()
    accepted, defaults = set(), {}
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        opts = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        accepted.update(opts)
        dest = next((o.lstrip("-").replace("-", "_") for o in opts if o.startswith("--")), None)
        for kw in node.keywords:
            if kw.arg == "default" and dest:
                defaults[dest] = kw.value.value if isinstance(kw.value, ast.Constant) else None
    assert accepted, f"no add_argument calls found in {rel_path} -- the AST reader has drifted"
    return accepted, defaults


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
    if module in SOURCE_ONLY:
        accepted, defaults = _parser_from_source(SOURCE_ONLY[module])
        how = "source"
    else:
        p = parser_for(module)
        accepted = {opt for action in p._actions for opt in action.option_strings}
        defaults = {a.dest: a.default for a in p._actions}
        how = "parser"

    required = harness_flags | EXTRA_ARGS[tag]
    if tag in ORACLE_BACKENDS:
        required |= {"--oracle_dataset"}
    if tag in NO_MANIFEST:
        required -= {"--manifest"}
    if tag in NO_OVERLAY:
        required -= {"--overlay"}
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
        f"[ok] {tag:<12} {len(accepted):>2} flags, capture_s={defaults['capture_s']}, "
        f"batch_size={defaults['batch_size']}  ({how})"
    )

if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print(f"\nall {len(DRIVERS)} offline drivers accept the harness CLI, with their per-backend defaults intact")
