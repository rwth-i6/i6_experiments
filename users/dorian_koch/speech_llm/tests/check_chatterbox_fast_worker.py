"""Guard (GPU, chatterbox venv): the worker's batched path synthesizes real dialogues without ever calling
``model.generate``, and lays them out exactly as the per-utterance path would.

On three short dialogues (real model, real voices, a random-speaker alias), with ``ChatterboxTurboTTS.generate``
replaced by a function that raises:

  * ``process_dialogues_batched`` runs and writes every dialogue (captured, not written): one track per speaker,
    every turn's audio non-empty and within 0.15-1.5 s per word, each track as long as its last turn ends;
  * the per-utterance path's own layout code (``process_dialogue`` -> ``gen_conversation``), run with a ``generate``
    that returns silence of exactly the batched path's lengths, picks the same voices, the same gaps and so the
    same start / end of every turn: the batched path draws speakers and silences on the same per-dialogue seed in
    the same order;
  * the raising ``generate`` does fire on the per-utterance path (this guard can see it being taken).

``check_chatterbox_fast_path.py`` is the CPU half (job argv, hash, worker source).

    <chatterbox venv python> recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_chatterbox_fast_worker.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
K = SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm"
VOICES = SETUP / "work/i6_experiments/users/dorian_koch/jobs/io/AggregateFoldersJob.2XetebWr32eR/output/out_folders"
os.environ.setdefault("HF_HOME", "/hpcwork/p0023999/common_hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(K))  # the worker's sibling imports (chatterbox_fast, clip_codec), as when the job runs it
spec = importlib.util.spec_from_file_location("chatterbox_inference", K / "chatterbox_inference.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

DIALOGUES = [
    [{"speaker": "user", "text": "Hey, can you tell me how long it takes to boil an egg?"},
     {"speaker": "assistant", "text": "About nine minutes for a hard boiled egg, a bit less if you like it soft."},
     {"speaker": "user", "text": "Great, thanks."}],
    [{"speaker": "assistant", "text": "Good morning! What would you like to talk about today?"},
     {"speaker": "user", "text": "I was wondering why the sky looks blue."},
     {"speaker": "assistant", "text": "Sunlight scatters off air molecules, and blue light scatters the most."},
     {"speaker": "user", "text": "That makes sense, I never knew that."}],
    [{"speaker": "user", "text": "Quick question."}, {"speaker": "assistant", "text": "Sure, go ahead."}],
]  # fmt: skip


class _Generate:
    def __init__(self):
        self.lengths, self.calls = None, 0

    def __call__(self, model, text, **_kw):
        self.calls += 1
        if self.lengths is None:
            raise RuntimeError("model.generate called")
        return torch.zeros(1, self.lengths.pop(0))


def main():
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    fails = []
    gen = _Generate()
    ChatterboxTurboTTS.generate = lambda self, text=None, **kw: gen(self, text, **kw)
    W.SPEAKER_ALIAS = {"user": "user_voices/rng_a", "assistant": "assistant_voices/prompt_0_voice_3"}
    with torch.inference_mode():
        model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        W.chatterbox_fast.speed_s3gen(model)
        written = {}
        W._write_dialogue = lambda out_dir, i, sr, spk_dir, tracks, utts, s2p, pt: written.__setitem__(
            i, (tracks, utts, s2p)
        )  # noqa: E731
        examples = [(100 + k, [dict(t) for t in d], None) for k, d in enumerate(DIALOGUES)]
        W.process_dialogues_batched(model, examples, "cuda", str(VOICES), "/nonexistent", {})
        if gen.calls:
            fails.append(f"the batched path called model.generate {gen.calls} times")
        sr = model.sr
        for k, d in enumerate(DIALOGUES):
            if 100 + k not in written:
                fails.append(f"dialogue {100 + k} not written")
                continue
            tracks, utts, s2p = written[100 + k]
            if sorted(tracks) != sorted({t["speaker"] for t in d}) or len(utts) != len(d):
                fails.append(f"dialogue {100 + k}: tracks {sorted(tracks)}, {len(utts)} turns for {len(d)}")
            for u, t in zip(utts, d):
                sec, words = (u["end"] - u["start"]) / sr, len(t["text"].split())
                if not 0.15 * words <= sec <= 1.5 * words + 1.0:
                    fails.append(f"dialogue {100 + k}: {sec:.2f} s for {words} words ({t['text'][:40]!r})")
            for s, tr in tracks.items():
                if tr.shape[-1] != max(u["end"] for u in utts):
                    fails.append(f"dialogue {100 + k}: track {s} length {tr.shape[-1]} != last end")
            # the per-utterance layout with the batched lengths: same voices, gaps, starts
            gen.lengths = [u["end"] - u["start"] for u in utts]
            legacy = {}
            W._write_dialogue = lambda out_dir, i, sr_, spk_dir, tracks, utts, s2p, pt: legacy.__setitem__(
                i, (utts, s2p)
            )  # noqa: E731
            W.process_dialogue(model, [dict(t) for t in d], "cuda", str(VOICES), W.silence_length_sampler,
                               "/nonexistent", 100 + k)  # fmt: skip
            l_utts, l_s2p = legacy[100 + k]
            if l_s2p != s2p:
                fails.append(f"dialogue {100 + k}: voices {s2p} vs per-utterance {l_s2p}")
            got = [(u["start"], u["end"]) for u in utts]
            want = [(u["start"], u["end"]) for u in l_utts]
            if got != want:
                fails.append(f"dialogue {100 + k}: timeline {got} vs per-utterance {want}")
        gen.lengths = None
        try:
            W.process_dialogue(model, [dict(t) for t in DIALOGUES[2]], "cuda", str(VOICES), W.silence_length_sampler,
                               "/nonexistent", 200)  # fmt: skip
            fails.append("the per-utterance path ran without calling model.generate: this guard cannot see it")
        except RuntimeError as e:
            if "model.generate called" not in str(e):
                raise
    for f in fails:
        print("FAIL", f)
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
