"""Login-node sanity test for FisherToMoshiTrainData (no GPU, no Sisyphus context).

Exercises the REAL job methods (via a shim, since sisyphus overrides Job.__new__) on real Fisher
data, checking the things that a rename could silently break: role<->channel mapping, agent-only
alignments, markup cleaning, and the arrow row layout.
"""

import io
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")

from i6_experiments.users.dorian_koch.speech_llm.fisher_prep import (  # noqa: E402
    MONOLOGUE_SPEAKER,
    FisherToMoshiTrainData,
    _clean_fisher_text,
)

CUTSET_PATH = (
    "work/speech_llm/full_duplex/sis_recipe/datasets/cutset_jobs/"
    "PrepareFisherDatasetJob.B3nfAjGOSR4d/output/fisher_original_manifest.jsonl"
)
WAV_DIR_PATH = (
    "work/i6_experiments/users/dorian_koch/speech_llm/fisher_prep/"
    "FisherSphToWav.bb5Gjm6qeOg8/output/audio"
)


class FakeTkPath:
    """Stand-in for `tk.Path`, which needs a Sisyphus job context to construct."""

    def __init__(self, path):
        self.path = path

    def get_path(self):
        return self.path


# --- 1. transcript markup cleaning ------------------------------------------------------------
markup_cases = [
    ("and i generally prefer", "and i generally prefer"),
    ("[laughter] yeah [noise] okay", "yeah okay"),
    ("(( i think )) so", "i think so"),
    ("(( ))", ""),
    ("we watched t._v.", "we watched t.v."),
    ("[mn] uh huh [sigh]", "uh huh"),
]
for raw_text, expected in markup_cases:
    got = _clean_fisher_text(raw_text)
    assert got == expected, f"clean({raw_text!r}) = {got!r}, want {expected!r}"
print(f"[ok] markup cleaning: {len(markup_cases)}/{len(markup_cases)} cases")

# --- 2. window planning -------------------------------------------------------------------------
# Sisyphus overrides Job.__new__ for its hashing machinery, so exercise the methods on a plain shim
# carrying the same functions -- this tests the real logic, not a reimplementation.
JobShim = type(
    "JobShim",
    (),
    {k: v for k, v in FisherToMoshiTrainData.__dict__.items() if not k.startswith("__")},
)
job = JobShim()
job.cutset = FakeTkPath(CUTSET_PATH)
job.wav_dir = FakeTkPath(WAV_DIR_PATH)
job.duration_sec = 40.0
job.stride_sec = 40.0
job.min_agent_speech_sec = 4.0
job.min_agent_words = 8
job.windows_per_conv = 2
job.max_windows = 40
job.seed = 0
job.rqmt = {"cpu": 4}

windows, stats = job._plan_windows()
print("[ok] planning:", stats)
assert len(windows) == 40, len(windows)
assert 0 < stats["conversations_used"] <= stats["conversations_total"], stats
# With windows_per_conv=2 and a 40-window cap, at most 40 conversations can have contributed --
# the old single `conversations` stat reported all 11699 here, wildly overstating diversity.
assert stats["conversations_used"] <= 40, stats

# --- 3. alignment invariants ---------------------------------------------------------------------
for window in windows:
    aligns = window["alignments"]
    assert aligns, window["conv"]
    assert all(0.0 <= a["start"] < job.duration_sec for a in aligns), window["conv"]
    assert all(a["end"] > a["start"] for a in aligns)
    assert all(a["start"] <= b["start"] for a, b in zip(aligns, aligns[1:]))
    assert all(a["speaker"] == MONOLOGUE_SPEAKER for a in aligns)
    for a in aligns:
        assert not set("[()_") & set(a["text"]), a
        assert a["text"] == a["text"].strip() and " " not in a["text"], a
words_per_window = [len(w["alignments"]) for w in windows]
print(
    f"[ok] alignments clean and in-window; words/window "
    f"min={min(words_per_window)} mean={sum(words_per_window) / len(words_per_window):.1f} "
    f"max={max(words_per_window)}"
)

# --- 4. channel mapping is role-keyed, not positional ---------------------------------------------
paths = job._channel_wav_paths("fe00001")
assert paths["user"].endswith("fe_03_00001A.wav"), paths
assert paths["agent"].endswith("fe_03_00001B.wav"), paths
print("[ok] channel mapping: user->A, agent->B")

# --- 5. audio slicing + row layout ----------------------------------------------------------------
row = job._build_row(windows[0])
assert set(row) == {"id", "duration", "audio_assistant", "audio_user", "alignments"}
for column in ("audio_assistant", "audio_user"):
    samples, sample_rate = sf.read(io.BytesIO(row[column]["bytes"]), dtype="float32")
    assert sample_rate == 8000, sample_rate
    assert samples.shape[0] == int(job.duration_sec * 8000), samples.shape
    rms = float(np.sqrt((samples**2).mean()))
    print(f"[ok] {column}: {samples.shape[0]} samples @ {sample_rate} Hz, rms={rms:.4f}")
    assert rms > 1e-4, f"{column} is silent"

# --- 6. the decisive check: alignments must line up with the ASSISTANT channel ---------------------
# If the A/B <-> user/agent mapping were inverted, the words would land on the silent stretches of
# the wrong track and this ratio would collapse to ~1.
samples, sample_rate = sf.read(io.BytesIO(row["audio_assistant"]["bytes"]), dtype="float32")
aligns = row["alignments"]
in_word_mask = np.zeros(samples.shape[0], dtype=bool)
for a in aligns:
    in_word_mask[int(a["start"] * sample_rate) : int(a["end"] * sample_rate)] = True
rms_in_word = float(np.sqrt((samples[in_word_mask] ** 2).mean()))
rms_out_of_word = float(np.sqrt((samples[~in_word_mask] ** 2).mean()))
ratio = rms_in_word / max(rms_out_of_word, 1e-9)
print(f"[ok] assistant rms in-word={rms_in_word:.4f} out-of-word={rms_out_of_word:.4f} ({ratio:.1f}x)")
assert ratio > 2.0, "alignments do not line up with assistant speech -- channel mapping suspect?"

print("\nid:", row["id"], "| words:", len(aligns))
print("monologue:", " ".join(a["text"] for a in aligns)[:300])
print("\nALL CHECKS PASSED")
