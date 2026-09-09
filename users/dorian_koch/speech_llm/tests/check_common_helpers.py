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
import tempfile
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


# --- chatterbox_benchmark_inference: the corpus write must STREAM, not accumulate -----------------
# This worker runs in the chatterbox venv, where i6_experiments is not importable, so it carries its
# own copy of clip_store's writer. Two bugs of the same shape have shipped in it. The copy boxed
# every sample as a Python float (~8x the array, and it makes datasets infer float64); and the
# writer held every clip in a list before handing the whole set to Dataset.from_dict, which copies
# it again into arrow -- three live copies of the corpus. Together they OOM-killed the
# 12,000-prompt rehearsal corpus three times (2026-09-07/08/09), each time AFTER ~2 h of GPU work
# had already produced every clip, and each time the response was to double rqmt["mem"]: 16 -> 48
# -> 120, ending only when c23g refused 128 outright. Doubling could never work, because peak
# scaled with the corpus. The writer now spools each clip to disk and streams the spool into arrow.
# This checks it still does -- a regression here reads as an ordinary refactor and costs a day.
_n = len(failures)
_bench_src = (
    SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/chatterbox_benchmark_inference.py"
).read_text()

# Source half: the synthesis loop must SPOOL each clip. Match the CODE line, not the file text --
# the docstrings legitimately quote the old shapes to explain why they are gone.
_code = [ln.strip() for ln in _bench_src.splitlines() if not ln.strip().startswith("#")]
if not any("spool.append(" in ln for ln in _code):
    failures.append(
        "the storage='hf' branch no longer spools clips to disk. Accumulating them in memory is "
        "what made peak RAM scale with the corpus and OOM-killed the rehearsal job three times."
    )
if not any("Dataset.from_generator(" in ln for ln in _code):
    failures.append(
        "the corpus is no longer written with Dataset.from_generator -- from_dict materialises "
        "every clip a second time, which is half of the OOM."
    )

# Behavioural half: exec the real ClipSpool / clip_features / write_clips with `datasets` stubbed,
# so the generator handed to from_generator is CAPTURED and this check can drive it itself. That is
# what makes the laziness assertion below possible at all.
_captured = {}


class _FakeDataset:
    features = "derived-from-from_dict"

    @staticmethod
    def from_dict(rows):
        _captured["from_dict_cols"] = sorted(rows)
        return _FakeDataset()

    @staticmethod
    def from_generator(gen, features=None):
        _captured["gen"] = gen
        _captured["features"] = features
        return _FakeDataset()

    def save_to_disk(self, path):
        _captured["path"] = path


_marker = "#: Base seed"
if "class ClipSpool:" not in _bench_src or _marker not in _bench_src:
    failures.append("could not locate ClipSpool/write_clips in the worker source")
else:
    _src = "class ClipSpool:" + _bench_src.split("class ClipSpool:")[1].split(_marker)[0]
    _ns = {
        "__name__": "bench_worker_under_test",
        "np": np,
        "os": os,
        "Dataset": _FakeDataset,
        "COL_INDEX": "index",
        "COL_AUDIO": "audio",
        "COL_SR": "sampling_rate",
        "COL_MONOLOGUE": "monologue",
        "COL_TRACE": "trace_json",
    }
    try:
        exec(compile(_src, "clip_spool", "exec"), _ns)
        with tempfile.TemporaryDirectory() as _td:
            _path = os.path.join(_td, "clips.f32")
            # deliberately RAGGED: the spool addresses clips by (offset, count), so equal-length
            # fixtures would pass even if the offsets were wrong.
            _clips = [np.random.default_rng(i).standard_normal(11 + 5 * i).astype(np.float32) for i in range(4)]
            _spool = _ns["ClipSpool"](_path)
            for _i, _c in enumerate(_clips):
                _spool.append(_i, _c)

            # A duplicate index must be refused: downstream joins address clips BY INDEX, so one
            # clip is silently dropped and every row after it misaligns.
            try:
                _spool.append(0, _clips[0])
                failures.append("ClipSpool accepted a duplicate clip index -- downstream joins misalign")
            except ValueError:
                pass

            # The spool must be ONE file. A file per clip would spend one inode per prompt on the
            # volume whose inode limit is the entire reason storage='hf' exists.
            _ns["write_clips"](os.path.join(_td, "out"), _spool, 24000)
            if len([f for f in os.listdir(_td) if f != "out"]) != 1:
                failures.append(f"the spool is not a single file: {sorted(os.listdir(_td))}")

            _rows = list(_captured["gen"]())
            if len(_rows) != len(_clips):
                failures.append(f"streamed {len(_rows)} rows for {len(_clips)} clips")
            else:
                for _i, (_row, _c) in enumerate(zip(_rows, _clips)):
                    _a = _row["audio"]
                    if not isinstance(_a, np.ndarray) or _a.dtype != np.float32:
                        failures.append(f"clip {_i} reached arrow as {type(_a).__name__}/{getattr(_a, 'dtype', '?')}")
                    elif not np.array_equal(_a, _c):
                        failures.append(
                            f"clip {_i} is not bit-exact through the spool ({_a.size} vs {_c.size} samples) "
                            f"-- the corpus would silently change"
                        )
                    if _row["index"] != _i or _row["sampling_rate"] != 24000:
                        failures.append(f"row {_i} carries index={_row['index']} sr={_row['sampling_rate']}")

            # LAZINESS, the whole point. Rewrite the spool AFTER write_clips returned: a streaming
            # generator reads per row and must see the new bytes, while any implementation that had
            # slurped the file up front would still hand back the old ones.
            _mutated = [c + 1.0 for c in _clips]
            with open(_path, "wb") as _fh:
                for _c in _mutated:
                    _fh.write(_c.tobytes())
            _after = [r["audio"] for r in _captured["gen"]()]
            if not all(np.array_equal(a, m) for a, m in zip(_after, _mutated)):
                failures.append(
                    "write_clips read the spool eagerly -- it materialises the corpus before arrow "
                    "sees it, which is the OOM this rewrite removed"
                )

            # The arrow schema must be DERIVED from the from_dict path it replaced, not hand-written
            # (a hand-written float64 here would silently double every clip on disk).
            if "from_dict_cols" not in _captured:
                failures.append("clip_features() no longer derives the schema from Dataset.from_dict")
        ok("clip write   streams through a 1-inode spool, bit-exact float32, lazily, dup-index refused", _n)
    except Exception as exc:
        failures.append(f"could not exercise chatterbox_benchmark_inference ClipSpool/write_clips: {exc!r}")

# --- clip_store.write_clips: the two writers of this format must agree on dtype -------------------
# clip_store is what the worker copy says it mirrors, but it appended `arr.tolist()`: Python floats,
# so datasets inferred float64 and every clip took twice the disk it needed -- while the worker
# wrote float32. A format with two writers that disagree on dtype is a format nobody can reason
# about, and neither reader complains because open_clips casts on the way out.
_n = len(failures)
_cs_src = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/clip_store.py").read_text()
_cs_appends = [
    ln.strip() for ln in _cs_src.splitlines() if "rows[COL_AUDIO].append(" in ln and not ln.strip().startswith("#")
]
if len(_cs_appends) != 1:
    failures.append(f"expected exactly one rows[COL_AUDIO].append in clip_store, found {_cs_appends}")
elif "tolist" in _cs_appends[0]:
    failures.append(
        f"clip_store.write_clips appends audio as {_cs_appends[0]!r}. .tolist() boxes every sample as "
        f"a Python float (~8x the array in RAM) and makes datasets infer float64, disagreeing with "
        f"the float32 the worker copy writes for the same format."
    )
else:
    ok("clip_store   writes float32 arrays, agreeing with the worker copy it mirrors", _n)

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
