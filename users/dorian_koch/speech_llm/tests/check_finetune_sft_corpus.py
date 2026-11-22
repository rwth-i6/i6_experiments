"""Guard: ``SpeechFinetune.write_config`` sizes and length-checks a run on the SFT dialogue corpus.

The SFT corpus (``sft_pipeline``: ``chatterbox_v2_inference`` rows, merged per cell by ``HfMergeShards``) has no
``duration`` column; ``resolve_max_steps`` read one from every corpus and the Haru instruction-SFT early read died in
``write_config`` (``Column 'duration' doesn't exist``). Haru's reader also cuts a row at ``hparams["max_s"]``, not at
``duration_sec``, so the truncation check must use that cut.

On a tiny corpus written by the real row writer (``chatterbox_v2_inference.make_row`` + ``FEATURES``):

  1. ``corpus_durations`` = the decoded FLAC track lengths (the last turn's end, without decoding audio).
  2. ``write_finetune_config`` with ``HARU_LIB_ADAPTER`` and a 150 s + 40 s corpus at ``max_s`` 300 (``duration_sec``
     at its default 100): writes the config with the given ``max_steps``. CAN FAIL: the code before 2026-11-22
     raises ``Column 'duration' doesn't exist``.
  3. The same at ``max_s`` 100: refused (half the rows longer than the read keeps).
  4. A corpus with a ``duration`` column is still read from it (``num_epochs`` sizing); one with neither is refused.

Setup venv, CPU, a few seconds::

    .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_finetune_sft_corpus.py
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the TTS job's flat imports
# the row writer's module imports the TTS packages (not in this venv); stubbed as check_tts_bounded does
for _name in ("chatterbox_fast", "chatterbox_fast_mtl", "chatterbox", "chatterbox.tts_turbo"):
    _m = types.ModuleType(_name)
    _m.ChatterboxTurboTTS = None
    sys.modules.setdefault(_name, _m)

import io  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import Audio, Dataset, Sequence, Value  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import finetune  # noqa: E402
from chatterbox_v2_inference import FEATURES, make_row  # noqa: E402

SR = 24000
FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


class _P:
    def __init__(self, p):
        self.p = p

    def get(self):
        return self.p

    def get_path(self):
        return self.p

    def __fspath__(self):  # tk.Path is os.PathLike: write_finetune_config opens out_config directly
        return self.p


def sft_corpus(root, lengths_s):
    """A corpus of dialogues with the given lengths, as the SFT pipeline writes them."""
    spk_dir = os.path.join(root, "voices")
    rows = []
    for i, n_s in enumerate(lengths_s):
        n = int(n_s * SR)
        tracks = {"assistant": np.zeros(n, np.float32), "user": np.zeros(n, np.float32)}
        tracks["user"][: SR // 2] = 0.1
        tracks["assistant"][n - SR // 2 :] = 0.1
        turns = [
            {"speaker": "user", "start_time": 0.0, "end_time": 0.5, "text": "hi", "lang": "en", "interrupted": False},
            {"speaker": "assistant", "start_time": n_s - 0.5, "end_time": float(n_s), "text": "hello", "lang": "en",
             "interrupted": False},
        ]  # fmt: skip
        p = {"id": str(i), "voices": {"assistant": os.path.join(spk_dir, "a.wav"), "user": os.path.join(spk_dir, "u.wav")},
             "exaggeration": {"assistant": 0.5, "user": 0.5}, "kind": "q3"}  # fmt: skip
        rows.append(
            make_row(p, tracks, SR, turns, [], [], spk_dir, "t_", {"system_prompt": "be nice"}, ["system_prompt"])
        )
    feats = FEATURES.copy()
    feats["system_prompt"] = Value("string")
    path = os.path.join(root, f"sft_{len(os.listdir(root))}")
    # as arrow, already encoded (the FLAC bytes): the TTS job's Audio() encode needs torchcodec, not in this venv
    Dataset(pa.Table.from_pylist(rows, schema=feats.arrow_schema)).save_to_disk(path)
    return path


def haru_job(root, train_data, max_s):
    job = type("FakeJob", (), {})()
    job.train_data, job.eval_data = train_data, None
    job.out_rundir = _P(os.path.join(root, f"run_{max_s}"))
    job.out_config = _P(os.path.join(root, f"config_{max_s}.yaml"))
    job.duration_sec, job.num_epochs, job.max_steps = 100, None, 300  # SpeechFinetune's default duration_sec
    job.audio_jitter_sec, job.lora_rank, job.seed = 0.0, 128, 0
    job.hparams = dict(voice_dir=_P(root), lr=1e-5, warmup_steps=30, grad_accum=8, audio_weight=1.0,
                       text_pad_weight=0.3, text_segment_gap=0.2, text_at="start", text_offset=-1, lead_in_frames=4,
                       max_s=float(max_s), save_every=50)  # fmt: skip
    job.adapter = finetune.HARU_LIB_ADAPTER
    return job


def main():
    root = tempfile.mkdtemp(prefix="check_ft_sft_")
    a, b = sft_corpus(root, [150.0, 40.0]), sft_corpus(root, [60.0])
    train_data = [(_P(a), 0.5), (_P(b), 0.5)]

    # 1. lengths without decoding = the decoded tracks
    durs = getattr(finetune, "corpus_durations", None)
    if durs is None:
        check(False, "1. finetune.corpus_durations exists")
    else:
        got = durs(a)
        sa = Dataset.load_from_disk(a).cast_column(
            "speaker_audio", {"speaker": Sequence(Value("string")), "audio": Sequence(Audio(decode=False))}
        )
        want = [max(sf.info(io.BytesIO(x["bytes"])).duration for x in r["speaker_audio"]["audio"]) for r in sa]
        check(np.allclose(got, want, atol=1e-3), f"1. corpus_durations {list(got)} == decoded track lengths {want}")

    # 2. the real caller on the SFT corpus, Haru's cut
    job = haru_job(root, train_data, 300)
    try:
        finetune.write_finetune_config(job, finetune.HARU_LIB_ADAPTER)
        cfg = open(job.out_config.get()).read()
        check("max_steps: 300" in cfg and a in cfg, "2. write_finetune_config on the SFT corpus at max_s 300")
    except Exception as e:  # noqa: BLE001
        check(False, f"2. write_finetune_config on the SFT corpus at max_s 300 raised {type(e).__name__}: {e}")

    # 3. the length check still bites at Haru's cut
    try:
        finetune.write_finetune_config(haru_job(root, train_data, 100), finetune.HARU_LIB_ADAPTER)
        check(False, "3. max_s 100 with a 150 s dialogue refused (it was accepted)")
    except ValueError as e:
        check("exceed duration_sec=100" in str(e), f"3. max_s 100 refused: {str(e)[:90]}")

    # 4. a corpus with a duration column; a corpus with neither
    d = os.path.join(root, "dur")
    Dataset.from_dict({"duration": [50.0, 90.0, 170.0]}).save_to_disk(d)
    steps = finetune.resolve_max_steps(train_data=_P(d), duration_sec=200, num_epochs=2, max_steps=1, batch_size=2)
    check(steps == 4, f"4a. duration column read (2 epochs x ceil(3 windows / 2) = 4 steps, got {steps})")
    x = os.path.join(root, "neither")
    Dataset.from_dict({"text": ["a"]}).save_to_disk(x)
    try:
        finetune.resolve_max_steps(train_data=_P(x), duration_sec=100, num_epochs=None, max_steps=1, batch_size=1)
        check(False, "4b. a corpus without lengths refused")
    except ValueError as e:
        check("not an SFT dialogue corpus" in str(e), f"4b. a corpus without lengths refused: {str(e)[:90]}")

    print(f"tmp dir {root}")
    if FAILS:
        raise RuntimeError(f"{len(FAILS)} check(s) failed: {FAILS}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
