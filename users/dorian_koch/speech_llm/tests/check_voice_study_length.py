"""Guard: a voice-sensitivity voice prompt is as long as the voice prompts training used (2026-10-12).

Bug: the voice study's ``know`` voice was the WHOLE 16.57 s ``prompt_0_voice_3.wav`` (208 code frames; upstream's
``load_voice_prompt`` does not cut) while training cut that same wav to 4.0 s / 50 frames (``encode_voice_prompt``),
so the published ``_voice_know`` rows were not under training conditions. Fix: ``know`` replays the v6 knowledge
rows' stored ``voice_codes`` (``PersonaPlexVoiceEmbeddings(codes=...)``); wav voices are checked at graph build.

  1. WORKER. The real ``voice_embeddings.stored_voice`` + ``embed`` (upstream's save path) on a tiny float64
     PersonaPlex LM: a 3-row codes dataset carrying one 50-frame voice -> ``<name>.pt`` with 49 embeddings (upstream's
     first step returns nothing) whose replay leaves the same streaming cache as stepping the codes directly; a
     dataset whose rows carry different voices raises.
  2. REAL DATA. ``stored_voice`` on the real v6 ``KnowledgeDialogueCodes`` dataset is ``[8, 50]`` = its summary's
     ``voice_n_frames`` = ``TRAIN_VOICE_SEC`` * 12.5, and ``TRAIN_VOICE_SEC`` == the ``voice_sec`` default of
     ``VoicePromptCodes`` and ``KnowledgeDialogueCodes``; the study's wav voices pass ``assert_wav_voice_lengths``;
     every non-NVIDIA voice is in exactly one of ``VOICE_WAVS`` (length-checked) and ``VOICE_CODES``; once ``voice_sensitivity/voices_codes`` exists its
     ``know.pt`` has 49 embeddings.
  3. CAN FAIL. (a) ``know`` declared as its wav again -> ``assert_wav_voice_lengths`` raises (208 > 50 frames);
     (b) a ``set_voice_codes`` that drops the last frame -> ``embed`` raises; (c) a ``stored_voice`` that takes row 0
     without comparing rows -> the mixed dataset is not caught (check 1 fails).

Setup .venv, CPU, ~1 min (from the setup root):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_voice_study_length.py
"""

import contextlib
import copy
import inspect
import json
import os
import sys
import tempfile
import types

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(ROOT, "recipe"))
sys.path.insert(0, os.path.join(ROOT, "recipe", "sisyphus"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

from moshi_family.personaplex import selfplay, voice_embeddings  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402

V6_CODES = os.path.join(ROOT, "output", "persona_knowledge", "v6_codes")
V6_SUMMARY = os.path.join(ROOT, "output", "persona_knowledge", "v6_codes_summary.json")
CODES_OUT = os.path.join(ROOT, "output", "voice_sensitivity", "voices_codes")

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
KW = dict(loaders._lm_kwargs)
KW.update(
    dim=64,
    num_heads=4,
    num_layers=2,
    hidden_scale=2.0,
    text_card=100,
    context=256,
    depformer_dim=32,
    depformer_dim_feedforward=64,
    depformer_num_heads=2,
    depformer_num_layers=1,
    dep_q=16,
)
BASE_LM = LMModel(device="cpu", dtype=torch.float64, **KW).eval()
STUB_MIMI = types.SimpleNamespace(reset_streaming=lambda: None)
FAILS = []


def check(ok, msg):
    print(("[ok] " if ok else "[FAIL] ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def make_gen():
    g = LMGen(
        copy.deepcopy(BASE_LM),
        device="cpu",
        use_sampling=False,
        audio_silence_frame_cnt=3,
        sample_rate=24000,
        frame_rate=12.5,
    )
    g.streaming_forever(1)
    return g


def codes_dataset(path, voices):
    rows = {
        "voice_codes": [v.reshape(-1).astype(np.int16).tolist() for v in voices],
        "voice_n_frames": [int(v.shape[1]) for v in voices],
    }
    Dataset.from_dict(rows).save_to_disk(path)
    return path


def worker_check(tmp, tag=""):
    """-> (pt frames, caches equal, mixed dataset raised)."""
    rng = np.random.default_rng(0)
    voice = rng.integers(0, 2048, size=(8, 50))
    ds = codes_dataset(os.path.join(tmp, f"ds{tag}"), [voice] * 3)
    v = voice_embeddings.stored_voice(ds)
    out = os.path.join(tmp, f"out{tag}")
    os.makedirs(out)
    with torch.no_grad():
        rep = voice_embeddings.embed(make_gen(), STUB_MIMI, out, "know", codes=v)
        replay = make_gen()
        replay.load_voice_prompt_embeddings(os.path.join(out, "know.pt"))
        replay.reset_streaming()
        replay._step_voice_prompt(STUB_MIMI)
        direct = make_gen()
        selfplay.set_voice_codes(direct, voice)
        direct.reset_streaming()
        direct._step_voice_prompt(STUB_MIMI)
    same = torch.equal(replay._streaming_state.cache, direct._streaming_state.cache)
    other = voice.copy()
    other[0, 0] = (other[0, 0] + 1) % 2048
    mixed = codes_dataset(os.path.join(tmp, f"mixed{tag}"), [voice, other, voice])
    try:
        voice_embeddings.stored_voice(mixed)
        raised = False
    except RuntimeError:
        raised = True
    return rep["frames"], same, raised


@contextlib.contextmanager
def patched(mod, name, fn):
    old = getattr(mod, name)
    setattr(mod, name, fn)
    try:
        yield
    finally:
        setattr(mod, name, old)


with tempfile.TemporaryDirectory() as tmp:
    # 1. worker
    frames, same, raised = worker_check(tmp)
    check(frames == 49, f"1a 50 stored code frames -> {frames} embeddings (want 49)")
    check(same, "1b replaying the .pt leaves the cache that stepping the stored codes leaves")
    check(raised, "1c a dataset whose rows carry different voices raises")

    # 2. real data + recipe
    from i6_experiments.users.dorian_koch.speech_llm.persona_knowledge import KnowledgeDialogueCodes
    from i6_experiments.users.dorian_koch.speech_llm.persona_prompts import VoicePromptCodes
    from speech_llm.full_duplex.sis_recipe.doriank import track_persona_voice as tpv

    defaults = {
        c.__name__: inspect.signature(c.__init__).parameters["voice_sec"].default
        for c in (VoicePromptCodes, KnowledgeDialogueCodes)
    }
    check(
        all(d == tpv.TRAIN_VOICE_SEC for d in defaults.values()),
        f"2a TRAIN_VOICE_SEC {tpv.TRAIN_VOICE_SEC} == {defaults}",
    )
    v6 = voice_embeddings.stored_voice(V6_CODES)
    with open(V6_SUMMARY) as f:
        want = json.load(f)["voice_n_frames"]
    check(
        v6.shape == (8, want) and want == round(tpv.TRAIN_VOICE_SEC * tpv.FRAME_RATE),
        f"2b real v6 voice {v6.shape}, summary voice_n_frames {want}",
    )
    wavs = {k: tpv.VOICE_WAVS_JOB[k] for k in tpv.VOICE_WAVS}
    try:
        tpv.assert_wav_voice_lengths(wavs)
        check(
            True,
            "2c wav voices within training's length: "
            + ", ".join(f"{k} {tpv.wav_code_frames(p)} frames" for k, p in wavs.items()),
        )
    except RuntimeError as e:
        check(False, f"2c {e}")
    custom = set(tpv.VOICE_WAVS_JOB)
    check(
        custom == set(tpv.VOICE_WAVS) | set(tpv.VOICE_CODES) and not set(tpv.VOICE_WAVS) & set(tpv.VOICE_CODES),
        f"2d custom voices {sorted(custom)} = VOICE_WAVS {tpv.VOICE_WAVS} + VOICE_CODES {tpv.VOICE_CODES}",
    )
    pt = os.path.join(CODES_OUT, "know.pt")
    if os.path.exists(pt):
        n = int(torch.load(pt, map_location="cpu")["embeddings"].shape[0])
        check(n == want - 1, f"2e landed {pt}: {n} embeddings (want {want - 1})")
    else:
        print(f"[skip] 2e {pt} not built yet", flush=True)

    # 3. mutants
    try:
        tpv.assert_wav_voice_lengths({"know": tpv.VOICE_WAVS_JOB["know"]})
        check(False, "3a know as its whole wav is NOT caught")
    except RuntimeError as e:
        check("208 frames" in str(e), f"3a know as its whole wav is caught: {e}")
    real_set = selfplay.set_voice_codes
    with patched(selfplay, "set_voice_codes", lambda g, c, name="x": real_set(g, np.asarray(c)[:, :-1], name=name)):
        try:
            worker_check(tmp, "_b")
            check(False, "3b a truncated replay is NOT caught")
        except RuntimeError as e:
            check(True, f"3b a truncated replay is caught: {e}")

    def lax(path):
        t = Dataset.load_from_disk(path)
        return np.asarray(t[0]["voice_codes"], dtype=np.int64).reshape(8, -1)

    with patched(voice_embeddings, "stored_voice", lax):
        _f, _s, raised_c = worker_check(tmp, "_c")
    check(not raised_c, "3c a stored_voice that does not compare rows misses the mixed dataset (check 1c fails)")

if FAILS:
    raise SystemExit(f"{len(FAILS)} check(s) failed: {FAILS}")
print("ALL OK", flush=True)
