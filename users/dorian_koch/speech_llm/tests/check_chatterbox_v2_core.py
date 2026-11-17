"""Guard: ChatterboxInferenceV2's model-free logic and job wiring (``chatterbox_v2_core``, ``tts.ChatterboxInferenceV2``).

Asserted against the real module and the real job:

  * language segments: inline ``<xx>...</xx>`` spans become segments, closing punctuation stays with the segment it
    closes, malformed tags raise; **no tag survives into a turn's text** (``render``'s ``turns.text``), which is what a
    model's text stream is built from;
  * the regeneration loop: a passing segment is synthesized once with its own seed (no regeneration = the same audio),
    a failing one again with new seeds up to ``max_attempts``, the best attempt is kept, and a segment too short to
    score (``exempt``) is never retried;
  * long segments are cut into TTS pieces at sentence ends under the length cap (an LLM-written 600-token turn ran an
    80 GB card out of memory), closing quotes staying with their sentence;
  * ``render``: segments of a turn are joined with ``segment_gap``, words land on the dialogue timeline inside their
    segment;
  * the job: its config is resolved at construction (hash pins it; ``rqmt`` / ``chunk`` are not hashed), it refuses
    an annotator for a language the backend does not have, and its argv carries both worker paths, which exist.

Each block is also run against a mutation that must make it fail (``MUTANTS``).

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_chatterbox_v2_core.py
"""

import os
import sys
from pathlib import Path

import numpy as np

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import job as sis_job  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import chatterbox_v2_core as core  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import tts  # noqa: E402

SR = 1000


def check_segments(c) -> list[str]:
    bad = []
    cases = {
        ("Das heißt <en>to commute</en> auf Englisch.", "de"): [("de", "Das heißt"), ("en", "to commute"),
                                                                ("de", "auf Englisch.")],
        ("这个词英语叫<en>commute</en>。你记住了吗？", "zh"): [("zh", "这个词英语叫"), ("en", "commute。"),
                                                       ("zh", "你记住了吗？")],
        ("<zh>你好</zh>", "en"): [("zh", "你好")],
        ("Just English.", "en"): [("en", "Just English.")],
    }  # fmt: skip
    for (text, lang), want in cases.items():
        got = [(s["lang"], s["text"]) for s in c.parse_segments(text, lang)]
        if got != want:
            bad.append(f"parse_segments({text!r}) = {got}, want {want}")
    for text in ("a <en>b", "a </en> b", "<en>a <de>b</de></en>", "<en>a</de>"):
        try:
            c.parse_segments(text, "de")
            bad.append(f"parse_segments({text!r}) did not raise")
        except ValueError:
            pass
    return bad


def check_regen(c) -> list[str]:
    errs = {"good": [0.0], "bad": [0.5, 0.3, 0.4], "fixed": [0.5, 0.1], "short": [1.0, 1.0, 1.0]}
    calls, seen = [], {}

    def synth(batch):
        calls.append([(b["text"], b["seed"]) for b in batch])
        return [b["text"] for b in batch]

    def check(wavs, batch):
        out = []
        for b in batch:
            k = seen.get(b["text"], 0)
            seen[b["text"]] = k + 1
            out.append({"err": errs[b["text"]][k], "exempt": b["text"] == "short"})
        return out

    items = [{"text": t, "seed": 100 + i} for i, t in enumerate(errs)]
    res = c.synthesize_checked(items, synth, check, lambda it: 0.2, 3)
    bad = []
    if calls[0] != [("good", 100), ("bad", 101), ("fixed", 102), ("short", 103)]:
        bad.append(f"attempt 0 must synthesize every item with its own seed: {calls[0]}")
    if [r["attempts"] for r in res] != [1, 3, 2, 1]:
        bad.append(f"attempts {[r['attempts'] for r in res]}, want [1, 3, 2, 1] (an exempt item is never retried)")
    if [r["err"] for r in res] != [0.0, 0.3, 0.1, 1.0]:
        bad.append(f"kept errors {[r['err'] for r in res]}, want the best attempt's [0.0, 0.3, 0.1, 1.0]")
    if len({s for call in calls for t, s in call if t == "bad"}) != 3:
        bad.append("each attempt of a failing item needs a new seed")
    return bad


def check_split(c) -> list[str]:
    """Long segments become TTS pieces cut at sentence ends (closing quotes stay with their sentence), none longer
    than the cap, none without letters, and the pieces join back to the text."""
    bad = []
    for text, lang in (('Er sagte: "Das ist ein sehr langer Satz, der immer weitergeht." ' * 6, "de"),
                       ("他说：“今天天气很好，我们一起去公园散步吧。”" * 6, "zh"), ("word " * 200, "en"), ("Short.", "en")):  # fmt: skip
        text = text.strip()
        ps = c.split_long(text, lang)
        cap = core.per_lang(core.MAX_ITEM_CHARS, lang)  # the real cap, whatever ``c`` is
        if any(len(p) > cap for p in ps):
            bad.append(f"split_long({lang}): a piece over {cap} characters")
        if any(not c._has_content(p) or p[0] in '"”」' for p in ps):
            bad.append(f"split_long({lang}): a piece without letters or starting with a closing quote: {ps}")
        if c.join_segments(ps).replace(" ", "") != text.replace(" ", ""):
            bad.append(f"split_long({lang}): pieces do not join back to the text")
    return bad


def check_render(c) -> list[str]:
    plan = {
        "turns": [{"speaker": "user", "text": "Wie sagt man <en>bike</en>?"}, {"speaker": "assistant", "text": "Gut."}],
        "segs": [[{"lang": "de", "text": "Wie sagt man"}, {"lang": "en", "text": "bike?"}], [{"lang": "de", "text": "Gut."}]],
        "items": [{"turn": 0, "segment": 0, "lang": "de", "text": "Wie sagt man"},
                  {"turn": 0, "segment": 1, "lang": "en", "text": "bike?"},
                  {"turn": 1, "segment": 0, "lang": "de", "text": "Gut."}],
        "gaps": [0.5, 0.0],
    }  # fmt: skip
    loud = np.ones(1000, np.float32) * 0.5
    seg = np.concatenate([np.zeros(200, np.float32), loud, np.zeros(300, np.float32)])  # 0.2 s silence, 1 s, 0.3 s
    res = [{"wav": seg, "err": 0.0, "attempts": 1, "asr_text": "x",
            "words": [{"text": "w", "start": 0.3, "end": 1.1}]} for _ in range(3)]  # fmt: skip
    tracks, turns, segs, words = c.render(plan, res, SR, 0.1)
    bad = []
    if [t["text"] for t in turns] != ["Wie sagt man bike?", "Gut."]:
        bad.append(f"turns.text {[t['text'] for t in turns]} (tags must be gone)")
    # turn 0: seg 0 keeps its lead (0.2 s), trail cut to 50 ms; gap 0.1; seg 1 lead cut to 50 ms, keeps its trail
    want0 = 0.2 + 1.0 + 0.05 + 0.1 + 0.05 + 1.0 + 0.3
    if abs(turns[0]["end_time"] - want0) > 2e-3:
        bad.append(f"turn 0 ends at {turns[0]['end_time']:.3f}, want {want0:.3f}")
    if abs(turns[1]["start_time"] - (want0 + 0.5)) > 2e-3:
        bad.append(f"turn 1 starts at {turns[1]['start_time']:.3f}, want {want0 + 0.5:.3f}")
    for w in words:
        s = segs[[k for k, x in enumerate(segs) if x["turn"] == w["turn"]][w["segment"]]]
        if not (s["start"] - 1e-6 <= w["start"] <= w["end"] <= s["end"] + 1e-6):
            bad.append(f"word {w} outside its segment {s}")
    if abs(words[1]["start"] - (segs[1]["start"] + 0.3 - 0.15)) > 2e-3:  # seg 1's lead was cut by 0.15 s
        bad.append(f"word in segment 1 not shifted by the trimmed lead: {words[1]} / {segs[1]}")
    if len(tracks["user"]) < int(want0 * SR) or set(tracks) != {"user", "assistant"}:
        bad.append("tracks")
    return bad


def check_job() -> list[str]:
    bad = []
    p = tk.Path("/dev/null")
    kw = dict(venv_python_path=p, annotate_python_path=p, parakeet_python_path=p, in_hf=tk.Path("/tmp/in"),
              speaker_dir=tk.Path("/tmp/voices"))  # fmt: skip

    def mk(**extra):
        sis_job.created_jobs.clear()  # same hash -> Sisyphus returns the cached object
        return tts.ChatterboxInferenceV2(**kw, **extra)

    j = mk()
    if j.config["annotators"] != {k: dict(v) for k, v in sorted(core.DEFAULT_ANNOTATORS.items())}:
        bad.append("config.annotators is not the resolved default")
    h = j._sis_id()
    if mk(rqmt={"gpu": 1, "mem": 64}, chunk=64)._sis_id() != h:
        bad.append("rqmt / chunk moved the hash")
    if mk(max_err={"*": 0.3})._sis_id() == h:
        bad.append("max_err did not move the hash")
    if mk(annotators={**core.DEFAULT_ANNOTATORS, "de": {"check": "qwen3", "words": "qwen3"}})._sis_id() == h:
        bad.append("annotators did not move the hash")
    for wrong in ({"*": {"check": "parakeet", "words": "parakeet"}, "zh": {"check": "parakeet", "words": "qwen3"}},
                  {"*": {"check": "whisper", "words": "parakeet"}}):  # fmt: skip
        try:
            mk(annotators=wrong)
            bad.append(f"accepted annotators {wrong}")
        except AssertionError:
            pass
    try:
        sis_job.created_jobs.clear()
        tts.ChatterboxInferenceV2(**{**kw, "parakeet_python_path": None})
        bad.append("accepted a parakeet annotator without parakeet_python_path")
    except AssertionError:
        pass
    here = Path(tts.__file__).resolve().parent
    for f in ("chatterbox_v2_inference.py", "speech_annotate_worker.py", "speech_text_norm.py", "chatterbox_v2_core.py",
              "chatterbox_inference.py", "chatterbox_fast.py", "chatterbox_fast_mtl.py"):  # fmt: skip
        if not (here / f).exists():
            bad.append(f"worker file {f} missing beside tts.py")
    return bad


def _mutant(**over):
    """``core`` with some functions replaced."""
    import types

    m = types.ModuleType("core_mutant")
    m.__dict__.update({k: v for k, v in core.__dict__.items() if not k.startswith("__")})
    m.__dict__.update(over)
    for fn in ("render", "synthesize_checked", "parse_segments", "turn_plain_text", "split_long", "_split_long"):
        if fn in over:
            continue
        f = m.__dict__[fn]
        if fn not in over:
            m.__dict__[fn] = types.FunctionType(f.__code__, m.__dict__, f.__name__, f.__defaults__)
    return m


MUTANTS = {
    "tags kept in turn text": (check_render, dict(plain_text=lambda t: t)),
    "no punctuation reattach / no tag errors": (check_segments, dict(TAG=__import__("re").compile(r"<(/?)(zz)>"))),
    "regeneration reuses the seed": (check_regen, dict(attempt_seed=lambda seed, attempt: seed)),
    "no inner trim": (check_render, dict(edge_trim=lambda wav, sr, **kw: (0, len(wav)))),
    "quotes cut off their sentence": (check_split, dict(_SENT_END=__import__("re").compile(r"[.!?。！？；;…]+\\s*"))),
    "no length cap": (check_split, dict(_split_long=lambda text, lang, max_chars: [text])),
}

if __name__ == "__main__":
    fails = check_segments(core) + check_split(core) + check_regen(core) + check_render(core) + check_job()
    for f in fails:
        print("FAIL", f)
    for name, (fn, over) in MUTANTS.items():
        if not fn(_mutant(**over)):
            fails.append(f"mutant {name!r} was not caught")
            print("FAIL", fails[-1])
        else:
            print(f"ok: mutant {name!r} caught")
    print(f"check_chatterbox_v2_core: {'FAIL' if fails else 'ok'}")
    sys.exit(1 if fails else 0)
