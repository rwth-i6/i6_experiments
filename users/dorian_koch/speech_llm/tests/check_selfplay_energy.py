"""Guard: self-play transcription records the LEVEL of the decoded audio, so "silence" cannot hide noise.

WHY. p0/p1 self-play read as 72-82% "silence" by ASR, but the audio was loud (-15 dBFS),
unintelligible babble with a looping text stream (review 2026-09-22). ASR words alone cannot tell the
two apart. ``selfplay_decode`` now measures each decoded channel (``channel_energy``: share of 80 ms
frames above -45 dBFS, integrated LUFS, RMS, peak) and ``SelfPlayTranscribe`` stores it as
``energy_a`` / ``energy_b``.

Drives the REAL ``SelfPlayTranscribe.run`` -> real ``selfplay_decode.main`` (in-process instead of a
subprocess) with a stub mimi whose decode returns designed audio per code stream, and a stub ASR
server. Channels: a -20 dBFS tone (ASR: words), digital silence, -15 dBFS noise (ASR: no words -- the
p0/p1 case), a -50 dBFS tone (below the -45 dBFS floor), a -40 dBFS tone (above it).

  1. Every channel's energy column matches its audio: tone/noise active ~1.0 with LUFS / RMS near the
     designed level, silence and -50 dBFS inactive, -40 dBFS active.
  2. The loud-but-wordless channel is told apart from the silent one: both have 0 ASR words, only one
     is active. With the energy columns absent (the pre-fix output) that distinction does not exist.
  3. A missing energy entry is an error, not an empty level.

Setup venv (torch, datasets, sphn, pyloudnorm), CPU, seconds:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_selfplay_energy.py
"""

import json
import os
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402
from sisyphus import tk  # noqa: E402

import huggingface_hub  # noqa: E402
import moshi_family.personaplex.selfplay_decode as spd  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.persona_selfplay as ps  # noqa: E402

SR, SEC = 24000, 4.0
N_SAMPLES = int(SR * SEC)
t = np.arange(N_SAMPLES) / SR
rng = np.random.default_rng(0)


def tone(dbfs):
    return np.sqrt(2) * 10 ** (dbfs / 20) * np.sin(2 * np.pi * 440 * t)


# code value (all entries of a stream) -> (decoded audio, ASR word count)
AUDIO = {
    1: (tone(-20), 10),
    2: (np.zeros(N_SAMPLES), 0),
    3: (np.clip(rng.normal(0, 10 ** (-15 / 20), N_SAMPLES), -1, 1), 0),
    4: (tone(-50), 0),
    5: (tone(-40), 2),
}
ROWS = [(1, 2), (3, 4), (5, 1)]  # (side a, side b) per conversation


class StubMimi:
    sample_rate = SR

    def decode(self, codes):
        return torch.as_tensor(AUDIO[int(codes.reshape(-1)[0])][0], dtype=torch.float32).view(1, 1, -1)


loaders.get_mimi = lambda *_a, **_k: StubMimi()
huggingface_hub.hf_hub_download = lambda *_a, **_k: "stub"

tmp = tempfile.mkdtemp(prefix="check_sp_energy_")
F, K = 50, 8
Dataset.from_list(
    [
        {
            "id": f"conv{i}",
            "n_codebooks": K,
            "n_frames": F,
            "frame_rate": 12.5,
            "codes_a": [a] * (K * F),
            "codes_b": [b] * (K * F),
            "text_a": "x",
            "text_b": "y",
        }
        for i, (a, b) in enumerate(ROWS)
    ]
).save_to_disk(os.path.join(tmp, "sp"))


def fake_run(cmd, **_kw):
    assert "moshi_family.personaplex.selfplay_decode" in cmd, cmd
    spd.main(cmd[cmd.index("--data") :])
    return subprocess.CompletedProcess(cmd, 0)


class FakeASR:
    """podcast_asr --serve: one JSON request per line, answers ok and writes the words file."""

    def __init__(self, *_a, **_k):
        self.stdin = self
        self.stdout = self
        self._last = None

    def write(self, line):
        req = json.loads(line)
        i, side = os.path.basename(req["wav"])[:-4].split("_")
        n = AUDIO[ROWS[int(i)][0 if side == "a" else 1]][1]
        with open(req["out"], "w") as f:
            json.dump({"words": [{"text": f"w{k}", "start": 0.2 * k, "end": 0.2 * k + 0.1} for k in range(n)]}, f)

    def flush(self):
        pass

    def readline(self):
        return json.dumps({"ok": True}) + "\n"

    def close(self):
        pass

    def wait(self):
        return 0


ps.subprocess = types.SimpleNamespace(run=fake_run, Popen=FakeASR, PIPE=None)
job = ps.SelfPlayTranscribe(
    selfplay_data=tk.Path(os.path.join(tmp, "sp")),
    venv_python_path=tk.Path("python"),
    asr_venv_python=tk.Path("python"),
)
job.out_dir = tk.Path(os.path.join(tmp, "out"))
cwd = os.getcwd()
os.chdir(tmp)
try:
    job.run()
finally:
    os.chdir(cwd)
out = load_from_disk(os.path.join(tmp, "out"))
assert {"energy_a", "energy_b", "words_a", "words_b"} <= set(out.column_names), out.column_names

# ---- 1 ----
got = {}
for i, r in enumerate(out):
    for side, code in zip("ab", ROWS[i]):
        got[code] = (json.loads(r[f"energy_{side}"]), len(json.loads(r[f"words_{side}"])))
e_tone, e_sil, e_noise, e_q50, e_q40 = (got[k][0] for k in (1, 2, 3, 4, 5))
assert e_tone["active_frac"] == 1.0 and abs(e_tone["rms_dbfs"] + 20) < 0.1, e_tone
assert e_tone["lufs"] is not None and abs(e_tone["lufs"] + 20) < 2.0, e_tone  # K-weighting ~0 dB at 440 Hz
assert e_noise["active_frac"] == 1.0 and abs(e_noise["rms_dbfs"] + 15) < 0.3, e_noise
assert e_sil["active_frac"] == 0.0 and e_sil["lufs"] is None and e_sil["active_median_db"] is None, e_sil
assert e_q50["active_frac"] == 0.0 and e_q40["active_frac"] == 1.0, (e_q50, e_q40)
print(
    f"[1] energy per channel: -20 dBFS tone active {e_tone['active_frac']:.2f} / {e_tone['lufs']:.1f} LUFS, "
    f"-15 dBFS noise active {e_noise['active_frac']:.2f}, silence 0.00, -50 dBFS 0.00, -40 dBFS 1.00"
)

# ---- 2 ----
(noise_e, noise_words), (sil_e, sil_words) = got[3], got[2]
assert noise_words == 0 and sil_words == 0, "fixture: both channels wordless by ASR"
assert noise_e["active_frac"] > 0.5 and sil_e["active_frac"] < 0.5, "loud wordless audio looks like silence"
old_cols = set(out.column_names) - {"energy_a", "energy_b"}
assert not any(c.startswith("energy") for c in old_cols), "without the columns nothing tells them apart"
print("[2] loud, wordless channel (0 ASR words, active 1.00) != silent channel (0 ASR words, active 0.00)")

# ---- 3 ----
d = tempfile.mkdtemp(prefix="check_sp_energy_missing_")
with open(os.path.join(d, "energy.jsonl"), "w") as f:
    f.write(json.dumps({"i": 0, "side": "a", **e_tone}) + "\n")
try:
    ps.energy_columns(d, 1)
    raise AssertionError("a missing energy entry was accepted")
except RuntimeError as e:
    assert "lacks 1 channels" in str(e), e
print("[3] a missing energy entry raises")
print("OK")
