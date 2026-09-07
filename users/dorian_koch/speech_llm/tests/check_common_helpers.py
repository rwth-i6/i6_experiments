"""Shared runtime helpers: `common.py`'s subprocess/port plumbing and the chatterbox worker's
reproducibility-critical samplers.

Migrated from the retired pytest suite `test_pipeline.py`, which was orphaned (referenced by
nothing, absent from this README) though still passing. Two of its cases were re-implementations
rather than guards -- they defined a local copy of the function under test, so they would have
passed no matter what the production code did. Both are now driven against the real code, which
required hoisting `silence_length_sampler` (chatterbox_inference) and
`strip_dialogue_markdown_fence` (hf_to_dialogue) out of the enclosing function bodies. The fence
check lives in `check_dialogue_templates.py`; the rest are here.

`chatterbox_inference.py` is a standalone worker script that imports torch / torchaudio /
chatterbox at module level, so it cannot be imported on a login node. We exec its real source with
those heavy modules stubbed -- the same "exec the real source" approach `check_launcher_config_reads`
uses -- so the assertions still run against the shipped text rather than a copy.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_common_helpers.py
"""

import os
import random
import socket
import sys
import types

import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))  # `sisyphus` package lives here
os.environ.setdefault("CUDA_HOME", "/usr")

failures = []


def ok(msg: str, since: int) -> None:
    """Print an [ok] line only if no failure was recorded since `since`.

    Printing [ok] unconditionally next to a recorded failure is how a broken guard reads as green.
    """
    if len(failures) == since:
        print(f"[ok] {msg}")


# --- common.py: pick_free_port ------------------------------------------------------------------
# Every managed server (vLLM judge, moshi server, unmute stack) picks its port through this. A port
# that is not actually bindable surfaces as a server that "never becomes ready" after a GPU has
# already been allocated.

from i6_experiments.users.dorian_koch.speech_llm.common import (  # noqa: E402
    pick_free_port,
    run_worker_script,
)

os.environ.pop("SLURM_JOB_ID", None)
_n = len(failures)
port = pick_free_port(18000)
if not isinstance(port, int):
    failures.append(f"pick_free_port returned {type(port).__name__}, not int")
elif not 18000 <= port <= 18000 + 999 + 50:
    failures.append(f"pick_free_port({18000}) returned {port}, outside the documented window")
else:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("localhost", port))
    except OSError as exc:
        failures.append(f"pick_free_port returned {port} but it is not bindable: {exc}")
    finally:
        probe.close()
ok(f"pick_free_port      {port} in range and bindable", _n)


# --- common.py: run_worker_script ---------------------------------------------------------------
# Every worker-script job goes through this. Non-str argv entries must be stringified (a bare int
# raises deep inside subprocess), and `with_hf_home=False` must genuinely omit HF_HOME -- a job that
# leaks the shared HF cache into a venv with a different datasets version fails at load time.

_n = len(failures)
captured = {}


def _fake_run(cmd, env=None, check=None):
    captured["cmd"] = cmd
    captured["env"] = env


from i6_experiments.users.dorian_koch.speech_llm import common  # noqa: E402

with patch.object(common.subprocess, "run", _fake_run):
    run_worker_script(
        "py",
        "/s.py",
        ["--a", 1, "--b", "x"],
        log_label="unit",
        with_hf_home=False,
        extra_env={"FOO": "bar"},
    )

if captured["cmd"] != ["py", "/s.py", "--a", "1", "--b", "x"]:
    failures.append(f"run_worker_script argv not stringified: {captured['cmd']}")
if captured["env"].get("PYTHONUNBUFFERED") != "1":
    failures.append("run_worker_script did not set PYTHONUNBUFFERED -- worker logs would buffer")
if captured["env"].get("FOO") != "bar":
    failures.append("run_worker_script dropped extra_env")
if "HF_HOME" in captured["env"]:
    failures.append("run_worker_script set HF_HOME despite with_hf_home=False")
ok("run_worker_script   argv stringified, env applied, HF_HOME suppressed", _n)


# --- TTS determinism (regression, 2026-08-03) ---------------------------------------------------
# Both chatterbox workers call a SAMPLING `model.generate`. They used to seed only Python `random`
# (for the speaker draw), leaving the audio itself unseeded -- so the benchmark's questions and the
# training corpus were different audio on every regeneration. Measured: two runs of the benchmark
# TTS on the same subsample gave 61 of 64 clips differing in LENGTH, worst per-sample diff 0.725
# (a PCM-16 step is 6.1e-5). That is an unmeasured noise floor under every knowledge number, and it
# made a storage-format A/B uninterpretable.
#
# The seed must be derived PER ITEM (clip index / dialogue id), not once per run, or a shard or a
# resumed run silently produces different audio for the same row. And it must come from hashlib,
# not the builtin hash(), which Python salts per process.
_n = len(failures)
for name, path in (
    (
        "chatterbox_benchmark_inference.py",
        SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_benchmark_inference.py",
    ),
    ("chatterbox_inference.py", SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_inference.py"),
):
    src = path.read_text()
    if "torch.manual_seed" not in src:
        failures.append(f"{name}: no torch.manual_seed -- the TTS sampling is unseeded again")
    if "hash(" in src.replace("hashlib", "").replace("_hash(", "") and "sha1" not in src:
        failures.append(f"{name}: derives a seed from builtin hash(), which is salted per process")
ben = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_benchmark_inference.py").read_text()
if "torch.manual_seed(SEED + i)" not in ben:
    failures.append("chatterbox_benchmark_inference.py: seed is not derived from the clip index")
conv = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_inference.py").read_text()
if "seed_for_dialogue" not in conv or "sha1" not in conv:
    failures.append("chatterbox_inference.py: no stable per-dialogue seed (seed_for_dialogue/sha1)")
ok("TTS determinism     both workers seed torch per item, from a stable (non-salted) hash", _n)


# --- chatterbox_benchmark_inference.write_clips: the deliberate copy must not drift ---------------
# This worker runs in the chatterbox venv, where i6_experiments is not importable, so it carries its
# own write_clips and SAYS it "mirrors clip_store". On 2026-09-07 it did not: clip_store used
# np.asarray, the copy used `[float(x) for x in samples]`, which boxes every audio sample as a Python
# float -- ~8x the array's memory, allocated for ALL clips at once just before Dataset.from_dict. The
# 12,000-prompt rehearsal corpus ran 2 h 09 m, jumped 11.3 -> 16.1 GB in ten seconds and was
# OOM-killed with nothing written. A copy that claims to mirror something has to be checked against it.
_n = len(failures)
_bench_src = (
    SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_benchmark_inference.py"
).read_text()

# Match the CODE line that appends audio, not the file text: the docstring legitimately quotes the
# old buggy expression to explain why it is gone, and a loose substring search flags that as the bug.
_audio_appends = [
    ln.strip() for ln in _bench_src.splitlines() if "rows[COL_AUDIO].append(" in ln and not ln.strip().startswith("#")
]
if len(_audio_appends) != 1:
    failures.append(f"expected exactly one rows[COL_AUDIO].append in the worker, found {_audio_appends}")
elif "np.asarray" not in _audio_appends[0]:
    failures.append(
        f"chatterbox_benchmark_inference.write_clips appends audio as {_audio_appends[0]!r}. It must "
        f"use np.asarray, as clip_store does: a Python-float comprehension there is ~8x the memory of "
        f"the float32 array, for every clip at once, and OOM-killed the 12,000-prompt rehearsal corpus."
    )
# Behavioural half: exec the real function (with `datasets` stubbed so from_dict/save_to_disk are
# captured rather than run) and assert what it hands to arrow is a float32 ARRAY, not a list.
_captured = {}


class _FakeDataset:
    @staticmethod
    def from_dict(rows):
        _captured["rows"] = rows
        return _FakeDataset()

    def save_to_disk(self, path):
        _captured["path"] = path


_ns = {
    "__name__": "bench_worker_under_test",
    "np": np,
    "Dataset": _FakeDataset,
    "COL_INDEX": "index",
    "COL_AUDIO": "audio",
    "COL_SR": "sampling_rate",
    "COL_MONOLOGUE": "monologue",
    "COL_TRACE": "trace_json",
}
_src = _bench_src.split("def write_clips(")[1]
_src = "def write_clips(" + _src.split("\n\n\n")[0]
try:
    exec(compile(_src, "write_clips", "exec"), _ns)
    _ns["write_clips"](
        "/tmp/unused", [(0, np.zeros(4, dtype=np.float32), 24000), (1, np.ones(4, dtype=np.float32), 24000)]
    )
    audio = _captured["rows"]["audio"]
    if not all(isinstance(a, np.ndarray) and a.dtype == np.float32 for a in audio):
        failures.append(
            f"write_clips handed arrow {[type(a).__name__ for a in audio]} instead of float32 arrays "
            f"-- a Python list here is the OOM"
        )
    # ...and a duplicate index must be refused, same rule as clip_store: downstream joins are BY index.
    try:
        _ns["write_clips"](
            "/tmp/unused", [(0, np.zeros(2, dtype=np.float32), 24000), (0, np.ones(2, dtype=np.float32), 24000)]
        )
        failures.append(
            "write_clips accepted a duplicate clip index -- one clip is silently dropped "
            "and every downstream join misaligns"
        )
    except ValueError:
        pass
    ok("write_clips  keeps float32 arrays (no Python-float boxing) and rejects duplicate indices", _n)
except Exception as exc:
    failures.append(f"could not exercise chatterbox_benchmark_inference.write_clips: {exc!r}")

# --- chatterbox_inference: reproducibility-critical helpers -------------------------------------

_HEAVY = ["torch", "torchaudio", "chatterbox", "chatterbox.tts_turbo", "datasets"]
_saved = {name: sys.modules.get(name) for name in _HEAVY}
for name in _HEAVY:
    stub = types.ModuleType(name)
    # MagicMock, not SimpleNamespace: the module body *calls* these (Features(...), Value("string")),
    # so every stubbed attribute has to be callable and chainable.
    stub.__getattr__ = lambda _attr: MagicMock()  # noqa: B023
    sys.modules[name] = stub

worker_src = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_inference.py").read_text()
worker = {"__name__": "chatterbox_inference_under_test", "__file__": "chatterbox_inference.py"}
try:
    exec(compile(worker_src, "chatterbox_inference.py", "exec"), worker)
except Exception as exc:  # pragma: no cover -- the stub set went stale
    failures.append(f"could not exec chatterbox_inference.py with stubbed deps: {exc!r}")
    worker = {}
finally:
    for name, mod in _saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod

# available_speakers must be sorted: the speaker assigned to a dialogue is drawn from this list by a
# seeded RNG, so if the order followed os.listdir (arbitrary, filesystem-dependent) the same seed
# would assign different voices on different machines and the corpus would not be reproducible.
_n = len(failures)
if "available_speakers" in worker:
    with patch("os.listdir", return_value=["z_voice.wav", "a_voice.wav", "m_voice.wav", "not_wav.mp3"]):
        speakers = worker["available_speakers"]("/fake/dir")
    if speakers != sorted(speakers):
        failures.append(f"available_speakers is not sorted ({speakers}) -- speaker draw is not reproducible")
    if speakers != ["a_voice", "m_voice", "z_voice"]:
        failures.append(f"available_speakers returned {speakers}; non-.wav files must be dropped")
    ok(f"available_speakers  sorted, non-wav dropped: {speakers}", _n)
else:
    failures.append("chatterbox_inference.available_speakers missing -- was it renamed?")

# The silence sampler is a truncated Gaussian: it resamples until the draw is inside the window.
# An out-of-range draw becomes a negative inter-turn gap large enough to reorder turns, or a long
# dead pause the model learns to imitate.
_n = len(failures)
if "silence_length_sampler" in worker:
    random.seed(42)
    lo, hi = worker["SILENCE_MIN"], worker["SILENCE_MAX"]
    draws = [worker["silence_length_sampler"]() for _ in range(5000)]
    out = [d for d in draws if not lo <= d <= hi]
    if out:
        failures.append(f"silence_length_sampler produced {len(out)} draw(s) outside [{lo}, {hi}]: {out[:3]}")
    # A sampler pinned to a constant would also pass the bounds check; require it to actually vary.
    if len(set(draws)) < len(draws) // 2:
        failures.append("silence_length_sampler is barely varying -- is it still sampling?")
    ok(f"silence sampler     {len(draws)} draws all within [{lo}, {hi}], varied", _n)
else:
    failures.append("chatterbox_inference.silence_length_sampler missing -- was it renamed or re-nested?")


# --- chatterbox_benchmark_inference: the BENCHMARK's voice ---------------------------------------
#
# Same bug as available_speakers above, in the sibling worker, missed because this guard only ever
# covered one of the two copies. `resolve_speaker_path` drew `random.choice(os.listdir(...))` over
# 128 user voices with NO sort, so which voice the knowledge benchmark asks its questions in was a
# property of the filesystem. It is stable on this cluster by luck -- the listing is verifiably not
# in sorted order.
#
# The fix is NOT to sort: with sorted() the same seed draws a different voice, which would silently
# re-voice the benchmark and make every existing knowledge number non-comparable. The historical
# choice is pinned instead, and sorting applies only to un-pinned aliases.

_HEAVY2 = ["torch", "torchaudio", "chatterbox", "chatterbox.tts_turbo", "datasets"]
_saved2 = {name: sys.modules.get(name) for name in _HEAVY2}
for name in _HEAVY2:
    stub = types.ModuleType(name)
    stub.__getattr__ = lambda _attr: MagicMock()  # noqa: B023
    sys.modules[name] = stub

bench_src = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_benchmark_inference.py").read_text()
bench = {"__name__": "chatterbox_benchmark_under_test", "__file__": "chatterbox_benchmark_inference.py"}
try:
    exec(compile(bench_src, "chatterbox_benchmark_inference.py", "exec"), bench)
except Exception as exc:  # pragma: no cover
    failures.append(f"could not exec chatterbox_benchmark_inference.py with stubbed deps: {exc!r}")
    bench = {}
finally:
    for name, mod in _saved2.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod

_n = len(failures)
if "resolve_speaker_path" in bench and "PINNED_RNG_SPEAKERS" in bench:
    resolve, pinned = bench["resolve_speaker_path"], bench["PINNED_RNG_SPEAKERS"]
    if pinned.get("user_voices/rng_a") != "prompt_9_prompt_0_voice_7.wav":
        failures.append(
            f"the pinned benchmark voice is {pinned.get('user_voices/rng_a')!r}; every knowledge "
            f"number to date was measured with 'prompt_9_prompt_0_voice_7.wav' (read off a finished "
            f"job's log). Changing it re-voices the benchmark and breaks comparability with ~30 "
            f"published tags -- if that is intended, it needs a new tag, not an edit here."
        )
    # Pinned alias: same answer whatever order the filesystem returns, including reversed.
    listing = [f"prompt_{i}_prompt_0_voice_{j}.wav" for i in range(13) for j in range(8)]
    got = set()
    for order in (listing, list(reversed(listing)), sorted(listing)):
        with patch("os.listdir", return_value=list(order)), patch("os.path.exists", return_value=True):
            got.add(resolve("/spk", "user_voices/rng_a"))
    if got != {"/spk/user_voices/prompt_9_prompt_0_voice_7.wav"}:
        failures.append(f"pinned rng_a resolved to {got} -- it must not depend on listing order")
    ok(f"benchmark voice     pinned, order-independent: {sorted(got)[0].split('/')[-1]}", _n)

    # A missing pinned voice must REFUSE, not quietly fall back to a fresh draw.
    _n = len(failures)
    with patch("os.listdir", return_value=list(listing)), patch("os.path.exists", return_value=False):
        try:
            resolve("/spk", "user_voices/rng_a")
            failures.append("a missing pinned speaker silently fell back to a draw -- it must raise")
        except AssertionError:
            pass
    ok("missing pin         refuses instead of re-voicing the benchmark", _n)

    # Un-pinned rng_ aliases must be sorted -- and the check is non-vacuous only if the unsorted
    # order would actually have given a different voice.
    _n = len(failures)
    picks = set()
    for order in (listing, list(reversed(listing))):
        random.seed(bench["SEED"])
        with patch("os.listdir", return_value=list(order)):
            picks.add(resolve("/spk", "user_voices/rng_b"))
    if len(picks) != 1:
        failures.append(f"un-pinned rng_ draw depends on listing order: {picks}")
    random.seed(bench["SEED"])
    unsorted_pick = "/spk/user_voices/" + random.choice(list(reversed(listing)))
    if unsorted_pick == next(iter(picks)):
        failures.append("fixture cannot distinguish sorted from unsorted -- the check proves nothing")
    ok("un-pinned rng_      sorted, order-independent", _n)
else:
    failures.append("chatterbox_benchmark_inference.resolve_speaker_path/PINNED_RNG_SPEAKERS missing")


if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print("\nshared runtime helpers OK")
