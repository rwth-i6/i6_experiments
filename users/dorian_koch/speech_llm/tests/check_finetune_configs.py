"""Guard: every finetune adapter renders a valid config, for both a single corpus and a mix.

``SpeechFinetune.write_config`` runs on the login node as a mini_task, so a renderer that crashes
(or silently emits a malformed value) costs a full manager round-trip to discover. Three real bugs
motivated this file:

  * three of the four renderers read ``job.train_data.get()`` directly, so passing the list-of-
    ``(path, weight)`` mix form raised ``AttributeError`` on everything except the base-Moshi lib
    adapter -- the one arm it had been developed against;
  * ``lr: 1e-06`` is parsed by YAML 1.1 as the *string* ``"1e-06"``, not a float (a float in
    exponent form needs both a decimal point and a signed exponent);
  * a ``{path: weight}`` dict silently loses the Sisyphus dependency edge, so the job starts before
    its corpus exists.

Run from the setup root, no GPU:
    .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_finetune_configs.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))  # `sisyphus` package lives here

import yaml  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.finetune import (  # noqa: E402
    MOSHI_ADAPTER,
    MOSHI_LIB_ADAPTER,
    MOSHI_LIB_FULL_ADAPTER,
    MOSHIRAG_LIB_ADAPTER,
    PERSONAPLEX_LIB_ADAPTER,
    PERSONAPLEX_LIB_LORA_ADAPTER,
    _yaml_float,
    train_data_specs,
)

ADAPTERS = {
    "moshi (fork schema)": MOSHI_ADAPTER,
    "moshi_lib": MOSHI_LIB_ADAPTER,
    "moshi_lib_full": MOSHI_LIB_FULL_ADAPTER,
    "personaplex_lib": PERSONAPLEX_LIB_ADAPTER,
    "personaplex_lib_lora": PERSONAPLEX_LIB_LORA_ADAPTER,
    "moshirag_lib": MOSHIRAG_LIB_ADAPTER,
}
# Only the base-Moshi lib launcher reads `train_data_mix`; the rest do cfg["train_data"]. The
# renderers must therefore REFUSE a mix for those, at render time, rather than emit a config their
# launcher dies on after the GPU has already been allocated.
SUPPORTS_MIX = {"moshi_lib", "moshi_lib_full"}  # same renderer, so the same mix support


class _FakePath:
    """Stand-in for a tk.Path: only ``.get()``/``.get_path()`` are used by the renderers."""

    def __init__(self, path):
        self.path = path

    def get(self):
        return self.path

    def get_path(self):
        return self.path


class _FakeJob:
    """Duck-typed SpeechFinetune: enough attributes for a renderer, no Sisyphus machinery."""

    def __init__(self, train_data, hparams=None, adapter=None):
        self.train_data = train_data
        self.eval_data = None
        self.out_rundir = _FakePath("/tmp/run_dir")
        self.out_config = _FakePath("/tmp/config.yaml")
        self.seed = 0
        self.duration_sec = 160
        self.audio_jitter_sec = 2.0
        self.lora_rank = 128
        self.hparams = hparams or {}
        #: A real SpeechFinetune always carries the adapter it is being rendered by, and the
        #: moshi_lib renderer reads it (`full_finetuning` is derived from `overlay_kind`, so the
        #: adapter is the single thing that decides LoRA vs full-FT). Omitting it here made this
        #: check pass against a job shape that cannot occur.
        self.adapter = adapter


SINGLE = _FakePath("/corpus/triviaqa")
MIX = [(_FakePath("/corpus/fisher_big"), 0.5), (_FakePath("/corpus/triviaqa_mix"), 0.5)]

# --------------------------------------------------------------------------------------------
# 1. Every adapter renders parseable YAML for a single corpus.
# --------------------------------------------------------------------------------------------
for label, adapter in ADAPTERS.items():
    job = _FakeJob(SINGLE, adapter=adapter)
    text = adapter.render_config(job, adapter.batch_size, 1500)
    cfg = yaml.safe_load(text)
    assert isinstance(cfg, dict) and cfg, f"{label}: config did not parse to a mapping"
    data = cfg.get("data", cfg)  # the fork schema nests under `data:`
    assert data.get("train_data") == "/corpus/triviaqa", (
        f"{label}: train_data not rendered ({data.get('train_data')!r})"
    )
    print(f"[ok] {label}: single-corpus config parses, train_data wired")

# --------------------------------------------------------------------------------------------
# 2. Adapters whose launcher understands a mix render `train_data_mix` correctly.
# --------------------------------------------------------------------------------------------
for label, adapter in ADAPTERS.items():
    if label not in SUPPORTS_MIX:
        continue
    cfg = yaml.safe_load(adapter.render_config(_FakeJob(MIX, adapter=adapter), adapter.batch_size, 1500))
    mix = cfg.get("train_data_mix")
    assert mix is not None, f"{label}: a mixed train_data produced no train_data_mix key"
    assert [row["path"] for row in mix] == ["/corpus/fisher_big", "/corpus/triviaqa_mix"], mix
    assert [row["weight"] for row in mix] == [0.5, 0.5], mix
    assert "train_data" not in cfg, f"{label}: emitted both train_data and train_data_mix"
    print(f"[ok] {label}: mixed-corpus config renders {len(mix)} weighted rows")

for label, adapter in ADAPTERS.items():
    if label in SUPPORTS_MIX:
        continue
    try:
        adapter.render_config(_FakeJob(MIX, adapter=adapter), adapter.batch_size, 1500)
    except AssertionError:
        print(f"[ok] {label}: refuses a mix its launcher cannot read")
        continue
    raise SystemExit(
        f"{label}: rendered a train_data mix, but its launcher only reads a single train_data key "
        f"-- the job would fail after the GPU is allocated"
    )

# --------------------------------------------------------------------------------------------
# 3. Numeric knobs must load as numbers, not strings.
# --------------------------------------------------------------------------------------------
job = _FakeJob(
    SINGLE,
    hparams={
        "lr": 1e-6,
        "depth_lr": 2e-6,
        "temporal_lr": 1e-6,
        "audio_other_weight": 0.02,
        "text_pad_weight": 0.3,
        "grad_accum": 32,
    },
    adapter=MOSHI_LIB_ADAPTER,
)
cfg = yaml.safe_load(MOSHI_LIB_ADAPTER.render_config(job, 16, 1500))
for key, expected in [
    ("lr", 1e-6),
    ("depth_lr", 2e-6),
    ("temporal_lr", 1e-6),
    ("audio_other_weight", 0.02),
    ("text_pad_weight", 0.3),
    ("grad_accum", 32),
    ("duration_sec", 160),
    ("audio_jitter_sec", 2.0),
]:
    got = cfg[key]
    assert isinstance(got, (int, float)) and not isinstance(got, bool), (
        f"{key} loaded as {type(got).__name__} ({got!r}) -- YAML 1.1 needs '1.0e-06', not '1e-06'"
    )
    assert abs(float(got) - expected) < 1e-12, f"{key}: {got} != {expected}"
print(f"[ok] all {8} numeric hparams load as numbers with the right values")

assert yaml.safe_load(f"x: {_yaml_float(1e-6)}")["x"] == 1e-6
assert isinstance(yaml.safe_load("x: 1e-06")["x"], str), (
    "if plain 1e-06 now parses as a float, PyYAML changed and _yaml_float's rationale is stale"
)
print("[ok] _yaml_float produces a YAML float where the bare literal produces a string")

# --------------------------------------------------------------------------------------------
# 4. train_data_specs rejects the dependency-losing dict form.
# --------------------------------------------------------------------------------------------
assert train_data_specs(SINGLE) == [(SINGLE, 1.0, None)]
assert [w for _, w, _ in train_data_specs(MIX)] == [0.5, 0.5]
# window_sec carries through: (path, weight, window_sec) 3-tuples for in-loader windowing.
assert train_data_specs([(SINGLE, 0.5, 160), (MIX[1][0], 0.5)]) == [(SINGLE, 0.5, 160.0), (MIX[1][0], 0.5, None)]
for bad, why in [
    ({SINGLE: 0.5}, "dict keyed by Path loses the Sisyphus dependency edge"),
    ([], "empty mix"),
    ([(SINGLE, 0.0)], "zero weight"),
    ([(SINGLE, -1.0)], "negative weight"),
]:
    try:
        train_data_specs(bad)
    except AssertionError:
        continue
    raise SystemExit(f"train_data_specs accepted a bad spec: {why}")
print("[ok] train_data_specs rejects the dict form, empty mixes and non-positive weights")

# --------------------------------------------------------------------------------------------
# 5. Adapter table is internally consistent.
# --------------------------------------------------------------------------------------------
names = [a.name for a in ADAPTERS.values()]
assert len(names) == len(set(names)), f"duplicate adapter names would collide in hashes: {names}"
for label, adapter in ADAPTERS.items():
    assert adapter.batch_size > 0, label
    assert adapter.launcher_module and adapter.pythonpath_package, label
    assert len(adapter.progress) == 2, label
print(f"[ok] {len(ADAPTERS)} adapters have unique names and complete fields")

# --------------------------------------------------------------------------------------------
# 6. PersonaPlex LoRA training-speed switches (2026-09-23): rendered only when an arm sets them.
#    p2 is RUNNING and p3 renders its config at start, so neither may see a new key; the preset
#    renders every key, with the right types, and adds nothing else.
# --------------------------------------------------------------------------------------------
from i6_experiments.users.dorian_koch.speech_llm.finetune import PERSONAPLEX_FAST_HPARAMS  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank.track_persona import (  # noqa: E402
    P2_HPARAMS,
    P2_TRAINED_HPARAMS,
)

PERF_KEYS = {
    "contiguous_step_inputs",
    "fused_causal_attention",
    "fused_lora_add",
    "depformer_skip_masked",
    "prefetch_batches",
    "length_bucket_pool",
}
ppx = PERSONAPLEX_LIB_LORA_ADAPTER
for label, hp in (("p2 (P2_TRAINED_HPARAMS)", P2_TRAINED_HPARAMS), ("p3 (P2_HPARAMS)", P2_HPARAMS)):
    assert not PERF_KEYS & set(hp), f"{label} carries a perf switch: {sorted(PERF_KEYS & set(hp))}"
    cfg = yaml.safe_load(ppx.render_config(_FakeJob(SINGLE, hparams=hp, adapter=ppx), ppx.batch_size, 3000))
    assert not PERF_KEYS & set(cfg), f"{label}: config renders {sorted(PERF_KEYS & set(cfg))}"
print("[ok] p2/p3 hparams carry no perf switch and render none")
# p2's config as it was written on 2026-09-23, before the switches existed: re-render it byte for byte.
P2_DIR = "work/i6_experiments/users/dorian_koch/speech_llm/finetune/SpeechFinetune.iAOoYovkkDAI/output"
if os.path.exists(os.path.join(P2_DIR, "config.yaml")):
    real = open(os.path.join(P2_DIR, "config.yaml")).read()
    job = _FakeJob(_FakePath(yaml.safe_load(real)["train_data"]), hparams=P2_TRAINED_HPARAMS, adapter=ppx)
    job.out_rundir = _FakePath(yaml.safe_load(real)["out_dir"])
    job.duration_sec = 60
    again = ppx.render_config(job, ppx.batch_size, 3000)
    assert again == real, "the current renderer no longer reproduces p2's config.yaml byte for byte"
    print("[ok] p2's config.yaml (written before the switches) re-renders byte for byte")
else:
    print(f"[--] {P2_DIR}/config.yaml not on disk; byte-for-byte p2 re-render skipped")
fast = yaml.safe_load(
    ppx.render_config(
        _FakeJob(SINGLE, hparams={**P2_HPARAMS, **PERSONAPLEX_FAST_HPARAMS}, adapter=ppx), ppx.batch_size, 3000
    )
)
plain = yaml.safe_load(ppx.render_config(_FakeJob(SINGLE, hparams=P2_HPARAMS, adapter=ppx), ppx.batch_size, 3000))
assert {k: fast[k] for k in PERSONAPLEX_FAST_HPARAMS} == PERSONAPLEX_FAST_HPARAMS, fast
assert {k: v for k, v in fast.items() if k not in PERSONAPLEX_FAST_HPARAMS} == plain, (
    "the preset changed a non-perf key"
)
assert "length_bucket_pool" not in PERSONAPLEX_FAST_HPARAMS, "bucketing changes the data order: not a speed-only preset"
bucket = yaml.safe_load(
    ppx.render_config(_FakeJob(SINGLE, hparams={"length_bucket_pool": 16}, adapter=ppx), ppx.batch_size, 10)
)
assert bucket["length_bucket_pool"] == 16 and isinstance(bucket["length_bucket_pool"], int)
print(f"[ok] PERSONAPLEX_FAST_HPARAMS renders {sorted(PERSONAPLEX_FAST_HPARAMS)} and changes nothing else")

print("\nALL CHECKS PASSED")
