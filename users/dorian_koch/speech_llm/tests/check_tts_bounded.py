"""Guard: ChatterboxInferenceV2's memory stays bounded in the number of dialogues, and a grouped job's audio does not
depend on its group (``chatterbox_v2_inference``, ``tts``), against the real modules:

  * the annotator calls (``Annotators.check`` / ``words``) send at most ``ANN_CALL`` transcribable clips per worker call
    (the SFT trial's English TTS job sent ~2,600 in one Parakeet call and was OOM-killed at 32 GB), and the sliced
    calls give the same results as one call: a stand-in worker that batches like the real ones (transcribable clips
    only, ``ASR_BATCH`` at a time, in order) and answers with each clip's batch, so a shifted batch boundary shows;
    clips at the 100 ms edge after Parakeet's resampling included;
  * ``seed_by_part``: a shard's dialogues get the same seeds alone and inside a group (``concat_group_inputs`` ->
    ``iter_chunks`` -> ``plan_dialogue``); without it the old seeds (the row index) stay; the job hashes as before
    when it is off.

Each block is also run against a mutation that must make it fail (``MUTANTS``).

Run from the setup root (~5 s):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_tts_bounded.py
"""

import json
import math
import os
import random
import sys
import tempfile
import types
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
HERE = Path(__file__).absolute().parent.parent  # speech_llm/
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("CUDA_HOME", "/usr")
for name in ("chatterbox_fast", "chatterbox_fast_mtl", "chatterbox", "chatterbox.tts_turbo"):
    m = types.ModuleType(name)
    m.ChatterboxTurboTTS = None
    sys.modules.setdefault(name, m)
try:
    import torchaudio  # noqa: F401
except ImportError:
    sys.modules["torchaudio"] = types.ModuleType("torchaudio")

import numpy as np  # noqa: E402

import chatterbox_v2_inference as inf  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import tts  # noqa: E402

SR = 24000


class _Worker:
    """Batches like ``parakeet_worker`` / ``speech_annotate_worker``: only transcribable clips, ``ASR_BATCH`` at a
    time, in order; each clip's answer names the clips of its batch (a moved batch boundary changes it)."""

    def __init__(self, min_len):
        self.min_len, self.calls = min_len, []

    def call_split(self, method, lists, **kw):
        pcms = kw["pcms"]
        idx = [i for i, p in enumerate(pcms) if self.min_len(len(p))]
        self.calls.append(len(idx))
        out = [[] if method != "transcribe_many" else "" for _ in pcms]
        for a in range(0, len(idx), inf.ASR_BATCH):
            part = idx[a : a + inf.ASR_BATCH]
            sig = "+".join(str(len(pcms[i])) for i in part)
            for i in part:
                out[i] = [{"text": f"{len(pcms[i])}@{sig}", "start": 0.0, "end": 1.0}] if method != "transcribe_many" \
                    else f"{len(pcms[i])}@{sig}"  # fmt: skip
        return out

    def call(self, method, **kw):  # score_many
        return [{"cer": 0.0, "wer": 0.0, "n_ref": 5} for _ in kw["refs"]]


def _parakeet_len(n):  # the worker's own rule: resampled to 16 kHz with torchaudio, >= 100 ms
    try:
        import torch
        import torchaudio.functional as F

        m = F.resample(torch.zeros(n), SR, inf.PARAKEET_SR).numel() if n else 0
    except Exception:
        m = math.ceil(n * inf.PARAKEET_SR / SR)
    return m >= int(0.1 * inf.PARAKEET_SR)


def _annotators(backend):
    a = inf.Annotators.__new__(inf.Annotators)
    a.cfg, a.metric = {"*": {"check": backend, "words": backend}}, "cer"
    a.parakeet = _Worker(_parakeet_len)
    a.annotate = _Worker(lambda n: n >= int(0.1 * SR))
    return a


def check_sliced_calls() -> list[str]:
    bad = []
    rng = random.Random(3)
    lens = [rng.choice([2398, 2399, 2400, 100, 0, 30000, 60000, 5000]) for _ in range(1300)]
    wavs = [np.zeros(n, np.float32) for n in lens]
    refs = ["x y z"] * len(wavs)
    for backend, lang, worker in (("parakeet", "en", "parakeet"), ("qwen3", "zh", "annotate")):
        a = _annotators(backend)
        got = a.check(wavs, SR, refs, lang)
        sliced_calls = list(getattr(a, worker).calls)
        b = _annotators(backend)
        old = inf.ANN_CALL
        try:
            inf.ANN_CALL = 10**9
            ref = b.check(wavs, SR, refs, lang)
        finally:
            inf.ANN_CALL = old
        if [g["asr_text"] for g in got] != [r["asr_text"] for r in ref]:
            bad.append(f"{backend}: sliced annotator calls give other results than one call")
        if max(sliced_calls) > inf.ANN_CALL or len(sliced_calls) < 2:
            bad.append(f"{backend}: a worker call got {max(sliced_calls)} clips (limit {inf.ANN_CALL})")
    a = _annotators("parakeet")
    if a.words(wavs, SR, refs, "en", [None] * len(wavs)) != _annotators("parakeet").words(wavs, SR, refs, "en", [None] * len(wavs)) \
            or max(a.parakeet.calls) > inf.ANN_CALL:  # fmt: skip
        bad.append("words: unbounded or unstable Parakeet calls")
    return bad


def _dlg(k):
    return {"lang": "de", "voices": {"user": "de/user/rng_v", "assistant": "de/assistant/rng_v"},
            "turns": [{"speaker": "user", "text": f"Hallo, Frage Nummer {k}?"},
                      {"speaker": "assistant", "text": "Gern, das erkläre ich."}]}  # fmt: skip


def _seeds(plan):
    return [it["seed"] for it in plan["items"]] + [round(g, 6) for g in plan["gaps"]] + sorted(plan["voices"].values())


def check_seed_by_part() -> list[str]:
    from datasets import Dataset, load_from_disk
    from sisyphus import job as sis_job
    from sisyphus import tk

    bad = []
    cfg = {"default_lang": "de", "english_model": "turbo", "code_switch": "segments", "tts_model": "qwen3"}
    with tempfile.TemporaryDirectory() as tmp:
        vd = Path(tmp) / "voices"
        for role in ("user", "assistant"):
            (vd / "de" / role).mkdir(parents=True)
            for v in range(6):
                (vd / "de" / role / f"v{v}.wav").write_bytes(b"")
        A = Dataset.from_list([{"dialogue": json.dumps(_dlg(k))} for k in range(5)])
        B = Dataset.from_list([{"dialogue": json.dumps(_dlg(100 + k))} for k in range(7)])
        A.save_to_disk(f"{tmp}/A")
        B.save_to_disk(f"{tmp}/B")

        def plans(paths, part):
            g = tts.concat_group_inputs([f"{tmp}/{p}" for p in paths], f"{tmp}/g_{''.join(paths)}")
            ds = load_from_disk(g)
            out = []
            for ch in inf.iter_chunks(ds, 0, [tts.GROUP_PART], 4, tts.GROUP_INDEX):
                for i, d, pt, key in ch:
                    if pt[tts.GROUP_PART] == str(part):
                        out.append(_seeds(inf.plan_dialogue(d, i, cfg, str(vd), key)))
            return out

        alone, grouped = plans(["A"], 0), plans(["B", "A"], 1)
        if alone != grouped or len(alone) != 5:
            bad.append("seed_by_part: a shard's seeds differ alone vs inside a group")
        ds = load_from_disk(f"{tmp}/A")
        old = [_seeds(inf.plan_dialogue(d, i, cfg, str(vd))) for ch in inf.iter_chunks(ds, 3, [], 4) for i, d, _, _ in ch]
        want = [_seeds(inf.plan_dialogue(json.loads(ds[k]["dialogue"]), 3 + k, cfg, str(vd))) for k in range(5)]
        if old != want:
            bad.append("without a seed key the seeds are no longer the row index's")
    sis_job.created_jobs.clear()
    kw = dict(venv_python_path=tk.Path("/x/py"), annotate_python_path=tk.Path("/x/an"), speaker_dir=tk.Path("/x/v"),
              parakeet_python_path=tk.Path("/x/pk"),
              in_hfs=[tk.Path("/x/a")], in_prefixes=["a_"])  # fmt: skip
    h0 = tts.ChatterboxInferenceV2(**kw)._sis_id()
    sis_job.created_jobs.clear()
    h1 = tts.ChatterboxInferenceV2(**kw, seed_by_part=False)._sis_id()
    sis_job.created_jobs.clear()
    h2 = tts.ChatterboxInferenceV2(**kw, seed_by_part=True)._sis_id()
    if h0 != h1 or h1 == h2:
        bad.append(f"job hash: off {h0}/{h1} (must match), on {h2} (must differ)")
    return bad


class _Patch:
    def __init__(self, obj, **over):
        self.obj, self.over, self.old = obj, over, {}

    def __enter__(self):
        for k, v in self.over.items():
            self.old[k] = getattr(self.obj, k)
            setattr(self.obj, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(self.obj, k, v)


_slices = inf.call_slices
_concat = tts.concat_group_inputs


def _concat_global(paths, out):  # mutant: the index over the whole group
    from datasets import concatenate_datasets, load_from_disk

    parts, n = [], 0
    for i, p in enumerate(paths):
        ds = load_from_disk(p)
        ds = ds.add_column(tts.GROUP_PART, [str(i)] * len(ds))
        parts.append(ds.add_column(tts.GROUP_INDEX, [str(n + k) for k in range(len(ds))]))
        n += len(ds)
    concatenate_datasets(parts).save_to_disk(out)
    return out


MUTANTS = {
    "one call per chunk": (check_sliced_calls, inf, dict(call_slices=lambda valid, per_call=0: [(0, len(valid))])),
    "slices by clip count, not transcribable clips": (check_sliced_calls, inf, dict(
        call_slices=lambda valid, per_call=None: [(a, min(a + inf.ANN_CALL, len(valid))) for a in range(0, len(valid), inf.ANN_CALL)])),
    "parakeet edge from the input rate": (check_sliced_calls, inf, dict(
        parakeet_valid=lambda pcms, sr: [len(p) >= int(0.1 * sr) for p in pcms])),
    "seed key = index in the whole group": (check_seed_by_part, tts, dict(concat_group_inputs=_concat_global)),
}  # fmt: skip

if __name__ == "__main__":
    fails = check_sliced_calls() + check_seed_by_part()
    for f in fails:
        print("FAIL", f)
    for name, (fn, obj, over) in MUTANTS.items():
        with _Patch(obj, **over):
            try:
                caught = bool(fn())
            except Exception as e:  # a mutant that crashes the check is caught too
                caught = True
                print(f"   ({name}: {type(e).__name__}: {e})")
        if not caught:
            fails.append(f"mutant {name!r} was not caught")
            print("FAIL", fails[-1])
        else:
            print(f"ok: mutant {name!r} caught")
    print(f"check_tts_bounded: {'FAIL' if fails else 'ok'}")
    sys.exit(1 if fails else 0)
