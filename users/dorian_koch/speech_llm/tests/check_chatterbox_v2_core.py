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
  * ``render``: every segment's edge silence is trimmed (a turn starts with speech; Multilingual V3 led with up to
    1 s of silence), segments of a turn are joined with a pause by punctuation (``boundary_pause``: a speaker's pause after a
    sentence or clause end, ``segment_gap`` inside a phrase), every turn is levelled to the same speech
    RMS (Multilingual V3 turns spanned 30 dB), words land on the dialogue timeline inside their segment;
  * pace: a segment off its speaking-rate band is regenerated and an in-pace attempt is preferred;
  * pauses: before a user turn they vary widely, before an assistant turn little;
  * interruptions (``<cut/>``): the whole turn is synthesized, cut after the last word before the marker, the next
    turn overlaps it, ``turns.text`` is what was said;
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


def check_pace(c) -> list[str]:
    """A segment whose check says ``pace_ok`` False is regenerated even at error 0, and an in-pace attempt beats an
    off-pace one with a lower error; ``speech_units`` counts words, Chinese characters."""
    bad = []
    pace = {"fast": [False, True], "slow": [False, False, False], "ok": [True]}
    errs = {"fast": [0.0, 0.05], "slow": [0.0, 0.0, 0.0], "ok": [0.0]}

    def synth(b):
        return [np.zeros(1, np.float32) for _ in b]

    def check(wavs, b):
        out = []
        for it in b:
            k = 0 if it["seed"] == it["base"] else 1 if it["seed"] == c.attempt_seed(it["base"], 1) else 2
            out.append({"err": errs[it["text"]][k], "pace_ok": pace[it["text"]][k], "k": k})
        return out

    items = [{"text": t, "seed": 7 + i, "base": 7 + i} for i, t in enumerate(("fast", "slow", "ok"))]
    res = c.synthesize_checked(items, synth, check, lambda it: 0.1, 3)
    if [r["attempts"] for r in res] != [2, 3, 1]:
        bad.append(f"pace: attempts {[r['attempts'] for r in res]}, want [2, 3, 1] (off-pace is regenerated)")
    if res[0]["k"] != 1:
        bad.append("pace: kept the off-pace attempt over the in-pace one with a higher error")
    if c.pace_check(1, 2.5, None, c.short_cap(1, "de"))["pace_ok"] or not c.pace_check(1, 0.6, None, c.short_cap(1, "de"))["pace_ok"]:
        bad.append("short cap: a one-word segment of 2.5 s (a repeat) passes, or one of 0.6 s fails")
    if (c.speech_units("Das ist gut, oder?", "de"), c.speech_units("你好，世界！", "zh")) != (4, 4):
        bad.append("speech_units: words / Chinese characters miscounted")
    return bad


def check_gaps(c) -> list[str]:
    """The default pauses: before a user turn they vary widely (some overlap, some > 1 s), before an assistant turn
    they stay in a narrow band; every draw inside its [lo, hi]."""
    import random

    bad, rng = [], random.Random(0)
    d = {sp: [c.draw_gap(rng, c.per_lang(c.DEFAULT_GAPS, sp)) for _ in range(2000)] for sp in ("user", "assistant")}
    u, a = np.array(d["user"]), np.array(d["assistant"])
    if not (u.std() > 3 * a.std() and (u < 0).any() and (u > 1.0).any()):
        bad.append(f"gaps: user std {u.std():.2f} (min {u.min():.2f}, max {u.max():.2f}) vs assistant {a.std():.2f}")
    for sp, x in d.items():
        lo, hi = c.per_lang(c.DEFAULT_GAPS, sp)[2:]
        if x and (min(x) < lo or max(x) > hi):
            bad.append(f"gaps: a {sp} draw outside [{lo}, {hi}]")
    return bad


def check_cut(c) -> list[str]:
    """An interrupted turn (``<cut/>``): the whole text is synthesized, the audio ends ``CUT_TAIL`` after the last word
    before the marker, later words are dropped, ``turns.text`` is what was said, and the next turn overlaps it."""
    bad = []
    t0 = {"speaker": "assistant", "lang": "de", "text": "Eins zwei drei <cut/> vier fünf."}
    if [x["text"] for x in c.turn_segments(t0, "de")] != ["Eins zwei drei vier fünf."]:
        bad.append(f"cut: segments {c.turn_segments(t0, 'de')} (the whole text is synthesized)")
    plan = {"lang": "de", "turns": [t0, {"speaker": "user", "text": "Moment!"}],
            "segs": [[{"lang": "de", "text": "Eins zwei drei vier fünf."}], [{"lang": "de", "text": "Moment!"}]],
            "items": [{"turn": 0, "segment": 0, "lang": "de", "text": "Eins zwei drei vier fünf."},
                      {"turn": 1, "segment": 0, "lang": "de", "text": "Moment!"}],
            "gaps": [-0.3, 0.0]}  # fmt: skip
    wav = np.full(3000, 0.3, np.float32)  # 3 s of sound, five words of 0.4 s every 0.5 s from 0.2 s
    ws = [{"text": w, "start": 0.2 + 0.5 * k, "end": 0.6 + 0.5 * k} for k, w in enumerate("eins zwei drei vier fünf".split())]
    res = [{"wav": wav, "err": 0.0, "attempts": 1, "asr_text": "", "words": ws},
           {"wav": np.full(500, 0.3, np.float32), "err": 0.0, "attempts": 1, "asr_text": "", "words": []}]  # fmt: skip
    tracks, turns, segs, words = c.render(plan, res, SR, 0.1, None)
    if turns[0]["text"] != "Eins zwei drei" or not turns[0]["interrupted"] or turns[1]["interrupted"]:
        bad.append(f"cut: turn rows {turns}")
    if [w["text"] for w in words if w["turn"] == 0] != ["eins", "zwei", "drei"]:
        bad.append(f"cut: words kept {[w['text'] for w in words if w['turn'] == 0]}")
    want_end = ws[2]["end"] + c.CUT_TAIL  # this wav has no silence to trim
    if abs(turns[0]["end_time"] - want_end) > 3e-3:
        bad.append(f"cut: turn 0 ends at {turns[0]['end_time']:.3f}, want {want_end:.3f}")
    if abs(turns[1]["start_time"] - (turns[0]["end_time"] - 0.3)) > 3e-3:
        bad.append("cut: the interrupting turn does not overlap the cut turn by its gap")
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
    def seg(amp):  # 0.2 s silence, 1 s at ``amp``, 0.3 s silence
        return np.concatenate([np.zeros(200, np.float32), np.full(1000, amp, np.float32), np.zeros(300, np.float32)])

    res = [{"wav": seg(a), "err": 0.0, "attempts": 1, "asr_text": "x",
            "words": [{"text": "w", "start": 0.3, "end": 1.1}]} for a in (0.5, 0.5, 0.02)]  # fmt: skip
    tracks, turns, segs, words = c.render(plan, res, SR, 0.1, -23.0)
    bad = []
    if [t["text"] for t in turns] != ["Wie sagt man bike?", "Gut."]:
        bad.append(f"turns.text {[t['text'] for t in turns]} (tags must be gone)")
    # every segment's lead and trail cut to 50 ms (a turn starts with speech: the pause before it is the gap alone)
    # and a language switch inside the turn keeps only INNER_MARGIN on each side (+ segment_gap)
    want0 = 0.05 + 1.0 + 0.02 + 0.1 + 0.02 + 1.0 + 0.05
    if abs(turns[0]["end_time"] - want0) > 2e-3:
        bad.append(f"turn 0 ends at {turns[0]['end_time']:.3f}, want {want0:.3f}")
    if abs(turns[1]["start_time"] - (want0 + 0.5)) > 2e-3:
        bad.append(f"turn 1 starts at {turns[1]['start_time']:.3f}, want {want0 + 0.5:.3f}")
    for w in words:
        s = segs[[k for k, x in enumerate(segs) if x["turn"] == w["turn"]][w["segment"]]]
        if not (s["start"] - 1e-6 <= w["start"] <= w["end"] <= s["end"] + 1e-6):
            bad.append(f"word {w} outside its segment {s}")
    for k, cut in ((0, 0.15), (1, 0.18)):  # each segment's lead was cut
        if abs(words[k]["start"] - (segs[k]["start"] + 0.3 - cut)) > 2e-3:
            bad.append(f"word in segment {k} not shifted by the trimmed lead: {words[k]} / {segs[k]}")
    # every turn levelled to -23 dBFS speech RMS: a 0.5 turn and a 0.02 turn (28 dB apart) end up equal
    for sp, t in (("user", turns[0]), ("assistant", turns[1])):
        x = tracks[sp][int(t["start_time"] * SR) : int(t["end_time"] * SR)]
        x = x[np.abs(x) > 1e-4]
        lvl = 20 * np.log10(np.sqrt(np.mean(x**2)))
        if abs(lvl + 23) > 0.5:
            bad.append(f"{sp} turn at {lvl:.1f} dBFS, want -23")
    if len(tracks["user"]) < int(want0 * SR) or set(tracks) != {"user", "assistant"}:
        bad.append("tracks")
    # a long silence inside one clip is shortened to CAPPED_PAUSE and the words after it move with the audio
    sil_wav = np.concatenate([np.full(500, 0.5, np.float32), np.zeros(2000, np.float32), np.full(500, 0.5, np.float32)])
    p3 = {"turns": [{"speaker": "user", "text": "x"}], "gaps": [0.0], "segs": [[{"lang": "de", "text": "Ja. Gut."}]],
          "items": [{"turn": 0, "segment": 0, "lang": "de", "text": "Ja. Gut."}]}  # fmt: skip
    r3 = [{"wav": sil_wav, "err": 0.0, "attempts": 1, "asr_text": "x",
           "words": [{"text": "Ja", "start": 0.0, "end": 0.5}, {"text": "Gut", "start": 2.5, "end": 3.0}]}]  # fmt: skip
    _, t3, _, w3 = c.render(p3, r3, SR, 0.1, None)
    if abs((t3[0]["end_time"] - t3[0]["start_time"]) - 1.35) > 0.02:
        bad.append(f"2 s inner silence not capped: turn lasts {t3[0]['end_time'] - t3[0]['start_time']:.2f} s, want 1.35")
    if abs(w3[1]["start"] - 0.85) > 0.02:
        bad.append(f"word after a capped pause at {w3[1]['start']:.2f}, want 0.85")
    # a boundary after a sentence end / a clause gets a speaker's pause, not segment_gap
    for first, want in (("Ja, genau.", 0.3), ("Ja, genau,", 0.12), ("Das heißt", 0.1)):
        p2 = {"turns": [{"speaker": "user", "text": "x"}], "gaps": [0.0],
              "segs": [[{"lang": "de", "text": first}, {"lang": "en", "text": "Okay."}]],
              "items": [{"turn": 0, "segment": 0, "lang": "de", "text": first},
                        {"turn": 0, "segment": 1, "lang": "en", "text": "Okay."}]}  # fmt: skip
        _, _, sg, _ = c.render(p2, res[:2], SR, 0.1, None)
        got = sg[1]["start"] - sg[0]["end"]  # segment rows include their margins
        if abs(got - want) > 2e-3:
            bad.append(f"pause after {first!r}: {got:.3f} s, want {want}")
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
    for fn in ("render", "synthesize_checked", "parse_segments", "turn_plain_text", "split_long", "_split_long",
               "level_to", "_rank", "turn_cut_units", "turn_plain_text", "turn_segments", "pace_check", "short_cap"):
        if fn in over:
            continue
        f = m.__dict__[fn]
        if fn not in over:
            g = types.FunctionType(f.__code__, m.__dict__, f.__name__, f.__defaults__)
            g.__kwdefaults__ = f.__kwdefaults__
            m.__dict__[fn] = g
    return m


MUTANTS = {
    "tags kept in turn text": (check_render, dict(plain_text=lambda t: t)),
    "no punctuation reattach / no tag errors": (check_segments, dict(TAG=__import__("re").compile(r"<(/?)(zz)>"))),
    "regeneration reuses the seed": (check_regen, dict(attempt_seed=lambda seed, attempt: seed)),
    "no inner trim": (check_render, dict(edge_trim=lambda wav, sr, **kw: (0, len(wav)))),
    "quotes cut off their sentence": (check_split, dict(_SENT_END=__import__("re").compile(r"[.!?。！？；;…]+\\s*"))),
    "no length cap": (check_split, dict(_split_long=lambda text, lang, max_chars: [text])),
    "turn lead kept": (check_render, dict(edge_trim=lambda wav, sr, lead, trail, **kw: (0 if lead else 0, len(wav)))),
    "no levelling": (check_render, dict(level_to=lambda wav, target, **kw: wav)),
    "pace ignored": (check_pace, dict(_rank=lambda r: (False, r["err"]))),
    "one pause distribution": (check_gaps, dict(DEFAULT_GAPS={"*": [0.2, 0.4, -0.3, 0.6]})),
    "interruption ignored": (check_cut, dict(turn_cut_units=lambda turn, lang: None)),
    "no short cap": (check_pace, dict(short_cap=lambda units, lang: 1e9)),
    "wide gap at a language switch": (check_render, dict(INNER_MARGIN=0.05)),
    "no pause at a sentence end": (check_render, dict(boundary_pause=lambda text, segment_gap: segment_gap)),
    "inner pauses kept": (check_render, dict(cap_pauses=lambda wav, sr, **kw: (wav, []))),
}

if __name__ == "__main__":
    fails = check_segments(core) + check_split(core) + check_regen(core) + check_pace(core) + check_gaps(core) + check_cut(core) + check_render(core) + check_job()
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
