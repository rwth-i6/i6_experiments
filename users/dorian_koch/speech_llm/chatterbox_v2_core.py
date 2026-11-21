"""The model-free core of ``ChatterboxInferenceV2`` (``chatterbox_v2_inference.py``): language segments, the annotator
config, the ASR-checked regeneration loop and segment joining. Pure Python + numpy, so the recipe, the dialogue
writer and the guards import it without the TTS venv, and the worker runs it as a sibling file.

**Language segments.** A turn is a list of ``{"lang", "text"}`` segments; one segment is a normal turn, several are a
turn that switches language. Writers mark a switch inline, ``Das heißt <en>to commute</en> auf Englisch.``: the
untagged text is in the turn's language, each ``<xx>...</xx>`` span in ``xx``. :func:`parse_segments` turns that into
segments (closing punctuation after a switch stays with the segment it closes), :func:`plain_text` removes the tags. The tags never reach a model's text stream: the TTS output's
``turns.text`` is the plain text, and the languages live in the separate ``segments`` column.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

#: an inline language tag: ``<en>``, ``</en>``, ``<zh>``, ``<pt-br>`` (lower-case language id)
TAG = re.compile(r"<(/?)([a-z]{2,3}(?:-[a-z]{2,4})?)>")

#: languages written without spaces between words (segments join without a space; scored per character)
CHAR_LANGS = frozenset({"zh", "ja", "yue"})

#: Chatterbox Multilingual's language ids (upstream SUPPORTED_LANGUAGES)
LANG_IDS = frozenset("ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh".split())

#: Parakeet TDT 0.6B v3's languages (its model card)
PARAKEET_V3_LANGS = frozenset("bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split())
#: Qwen3-ForcedAligner-0.6B's languages (its model card); Qwen3-ASR transcribes these and more
QWEN3_ALIGNER_LANGS = frozenset("zh en yue fr de it ja ko pt ru es".split())

#: Which backend checks a language's audio (``check``: transcript scored against the text, drives regeneration) and
#: which writes its word timings (``words``). Backends: ``parakeet`` (Parakeet v3; words = its transcript's words with
#: times), ``qwen3`` (check: Qwen3-ASR-0.6B; words: Qwen3-ForcedAligner-0.6B timing the GIVEN text, per character for
#: Chinese / Japanese), ``none``. Key ``*`` = every language not listed.
DEFAULT_ANNOTATORS = {
    "*": {"check": "parakeet", "words": "parakeet"},
    "zh": {"check": "qwen3", "words": "qwen3"},
    "ja": {"check": "qwen3", "words": "qwen3"},
    "yue": {"check": "qwen3", "words": "qwen3"},
}
BACKEND_LANGS = {"parakeet": PARAKEET_V3_LANGS, "qwen3": None, "none": None}  # None: checked by the backend itself

#: Regenerate a segment whose normalised character error rate is above this (``metric="cer"``, the default: graded on
#: long words and compounds, where one misheard ending costs a whole word in WER). Key ``*`` as above.
DEFAULT_MAX_ERR = {"*": 0.1}
#: A segment with fewer reference tokens is scored but never regenerated: one or two words are too coarse to judge
#: (a perfect German "Hallo" inside a Chinese turn is transcribed "Hello", error 1.0; seen 2026-11-17, 3 attempts).
DEFAULT_MIN_CHECK_TOKENS = 3

#: Speaking-rate bands per speaker role and language, in units per second of speech (words; characters for
#: ``CHAR_LANGS``), measured over the segment's sounding span (:func:`edge_trim`). A segment outside its band is
#: regenerated like one above ``max_err`` (a voice prompt sometimes makes the decoder rush, or babble for seconds
#: before the text: seen 2026-11-17, 17.8 s before the first word, CER 0). ``{}`` = no pace check.
DEFAULT_PACE: dict = {}
#: a segment with fewer units is never paced (one or two words have no stable rate)
DEFAULT_MIN_PACE_UNITS = 6


def speech_units(text: str, lang: str) -> int:
    """Words, or characters for ``CHAR_LANGS`` (letters and digits only): what a speaking rate counts."""
    if lang in CHAR_LANGS:
        return sum(1 for c in text if unicodedata.category(c)[0] in "LN")
    return sum(1 for w in text.split() if _has_content(w))


#: A segment too short to pace (one word, one character) must still not run long: Chatterbox repeats or babbles on
#: very short inputs ("Wasser?" came out as "Wasser, Wasser", single Chinese characters as several syllables;
#: 2026-11-17), which a transcript check exempted as too short to score. Longest sounding span: base + per unit.
SHORT_MAX_S = {"*": (0.8, 0.7), "zh": (0.6, 0.45), "ja": (0.6, 0.45), "yue": (0.6, 0.45)}


def short_cap(units: int, lang: str) -> float:
    base, per = per_lang(SHORT_MAX_S, lang)
    return base + per * max(units, 1)


def pace_check(units: int, speech_s: float, band, cap: float | None = None) -> dict:
    """``{"rate", "pace_ok"}``: ``units`` over ``speech_s`` seconds against ``band`` = ``(lo, hi)``, and ``speech_s``
    against ``cap`` (None: unchecked)."""
    rate = units / max(speech_s, 1e-3)
    ok = (band is None or band[0] <= rate <= band[1]) and (cap is None or speech_s <= cap)
    return {"rate": rate, "pace_ok": ok}


def per_lang(cfg: dict, lang: str):
    """``cfg[lang]``, else ``cfg["*"]``."""
    return cfg[lang] if lang in cfg else cfg["*"]


def check_annotators(cfg: dict) -> None:
    """Raise on an unknown backend or role (at job construction, not hours into a run)."""
    assert "*" in cfg, "annotators need a '*' entry (every language not listed)"
    for lang, roles in cfg.items():
        assert set(roles) == {"check", "words"}, f"annotators[{lang!r}]: roles must be exactly check, words"
        for role, b in roles.items():
            assert b in BACKEND_LANGS, f"annotators[{lang!r}][{role!r}]: unknown backend {b!r}"
            if b == "none" and role == "check":
                continue
            ok = BACKEND_LANGS[b]
            assert lang == "*" or ok is None or lang in ok, f"{b} does not support {lang!r}"


def _has_content(s: str) -> bool:
    return any(unicodedata.category(c)[0] in "LN" for c in s)


def _is_cjk(c: str) -> bool:
    return "぀" <= c <= "ヿ" or "㐀" <= c <= "鿿" or "豈" <= c <= "﫿" or "＀" <= c <= "￯"


def join_segments(texts: list[str]) -> str:
    """Segment texts as one turn text: a space between them unless a side is CJK or punctuation already spaces it."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out and not (
            _is_cjk(out[-1]) or _is_cjk(t[0]) or unicodedata.category(t[0])[0] == "P" and t[0] not in "(\"'«„“"
        ):
            out += " "
        out += t
    return out


def plain_text(text: str) -> str:
    """``text`` without language tags (what a model's text stream sees), whitespace collapsed."""
    return re.sub(r"\s+", " ", TAG.sub("", text)).strip()


def parse_segments(text: str, lang: str) -> list[dict]:
    """``[{"lang", "text"}]`` of a turn written in ``lang`` with inline ``<xx>...</xx>`` spans. Raises ``ValueError`` on
    nested, unclosed or mismatched tags. A piece without letters or digits (punctuation after a closing tag) joins the
    segment before it; neighbouring segments of one language merge."""
    segs, pos, cur = [], 0, None
    for m in TAG.finditer(text):
        closing, tag = m.group(1) == "/", m.group(2)
        segs.append((cur or lang, text[pos : m.start()]))
        pos = m.end()
        if closing:
            if cur != tag:
                raise ValueError(f"closing </{tag}> without an open <{tag}> in {text!r}")
            cur = None
        else:
            if cur is not None:
                raise ValueError(f"nested <{tag}> inside <{cur}> in {text!r}")
            cur = tag
    if cur is not None:
        raise ValueError(f"unclosed <{cur}> in {text!r}")
    segs.append((lang, text[pos:]))
    out: list[dict] = []
    for lg, t in segs:
        if out and lg != out[-1]["lang"]:  # closing punctuation right after a switch belongs to the segment before
            k = 0
            while k < len(t) and (t[k].isspace() or unicodedata.category(t[k]) in ("Po", "Pe", "Pf")):
                k += 1
            if t[:k].strip():
                out[-1]["text"] = join_segments([out[-1]["text"], t[:k]])
                t = t[k:]
        if not t.strip():
            continue
        if out and (not _has_content(t) or out[-1]["lang"] == lg):
            out[-1]["text"] = join_segments([out[-1]["text"], t])
        elif not _has_content(t):
            continue  # leading punctuation only
        else:
            out.append({"lang": lg, "text": t.strip()})
    return out


#: An interruption: ``<cut/>`` in a turn's text marks where the next speaker cuts in. The whole text is synthesized
#: (so the voice does not wind down before the cut), the audio is cut right after the last word before the marker, the
#: next turn starts overlapping it, and ``turns.text`` holds only what was said before the cut.
CUT = "<cut/>"


def _turn_text(turn: dict) -> str:
    text = turn["text"]
    return " ".join(str(t) for t in text) if isinstance(text, list) else text  # some writers emit a one-element list


def turn_segments(turn: dict, lang: str) -> list[dict]:
    """A dialogue turn's segments: its ``segments`` if given, else its ``text`` (without a ``CUT`` marker: the whole
    turn is synthesized) parsed in ``turn["lang"]`` or ``lang``."""
    if turn.get("segments"):
        return [{"lang": s["lang"], "text": s["text"].strip()} for s in turn["segments"] if _has_content(s["text"])]
    return parse_segments(re.sub(r"\s*" + re.escape(CUT) + r"\s*", " ", _turn_text(turn)), turn.get("lang") or lang)


def turn_cut_units(turn: dict, lang: str) -> int | None:
    """Speech units (:func:`speech_units`, per language segment) said before the turn's ``CUT``; None = no cut."""
    if turn.get("segments") or CUT not in _turn_text(turn):
        return None
    pre = _turn_text(turn).split(CUT)[0]
    tags = list(TAG.finditer(pre))
    if tags and tags[-1].group(1) != "/":  # cut inside a language span (full SFT run: 3 TTS jobs crashed on one)
        pre += f"</{tags[-1].group(2)}>"
    return sum(speech_units(s["text"], s["lang"]) for s in parse_segments(pre, turn.get("lang") or lang))


def turn_plain_text(turn: dict) -> str:
    """A turn's text as a model's text stream sees it: no language tags, and of an interrupted turn only what was said
    before the cut."""
    if turn.get("segments"):
        return join_segments([s["text"] for s in turn["segments"]])
    return plain_text(_turn_text(turn).split(CUT)[0])


#: longest text one TTS item gets (characters; CHAR_LANGS count each character). Longer segments are split at
#: sentence ends (then commas, then spaces): Chatterbox degrades on long inputs, and the batched decoders size their
#: static KV cache by the longest item (a 600-token LLM-written turn ran an 80 GB card out of memory, 2026-11-17).
MAX_ITEM_CHARS = {"*": 220, "zh": 90, "ja": 90, "yue": 90}
#: a sentence / clause end with the closing quotes and brackets that belong to it, and the space after it
_SENT_END = re.compile(r"[.!?。！？；;…]+[\"'”’»)\]）」』]*\s*")
_CLAUSE_END = re.compile(r"[,，、:：]+[\"'”’»)\]）」』]*\s*")


def _cut_after(pat, text: str) -> list[str]:
    parts, pos = [], 0
    for m in pat.finditer(text):
        parts.append(text[pos : m.end()].strip())
        pos = m.end()
    parts.append(text[pos:].strip())
    return [p for p in parts if p]


def split_long(text: str, lang: str, max_chars: dict = MAX_ITEM_CHARS) -> list[str]:
    """``text`` as pieces of at most ``max_chars[lang]`` characters, cut at sentence ends, else clause ends, else
    spaces (else hard), sentences packed greedily; ``join_segments(pieces)`` restores ``text`` up to whitespace. A
    piece without letters or digits (a closing quote after a sentence end) joins the piece before it."""
    out: list[str] = []
    for p in _split_long(text, lang, max_chars):
        if out and not _has_content(p):
            out[-1] = join_segments([out[-1], p])
        else:
            out.append(p)
    return out


def _split_long(text: str, lang: str, max_chars: dict) -> list[str]:
    n = per_lang(max_chars, lang)
    if len(text) <= n:
        return [text]
    out = []
    for pat in (_SENT_END, _CLAUSE_END):
        parts = _cut_after(pat, text)
        if len(parts) > 1:
            cur = ""
            for p in parts:
                if cur and len(cur) + 1 + len(p) > n:
                    out.append(cur)
                    cur = p
                else:
                    cur = join_segments([cur, p]) if cur else p
            out.append(cur)
            return [q for p in out for q in _split_long(p, lang, max_chars)] if any(len(p) > n for p in out) else out
    words = text.split(" ")
    if len(words) > 1:
        cut = max(1, len(words) // 2)
        return _split_long(" ".join(words[:cut]), lang, max_chars) + _split_long(" ".join(words[cut:]), lang, max_chars)
    return [text[i : i + n] for i in range(0, len(text), n)]


def main_lang(segments: list[dict]) -> str:
    """The language with the most characters (a turn's ``lang``)."""
    n: dict[str, int] = {}
    for s in segments:
        n[s["lang"]] = n.get(s["lang"], 0) + len(s["text"])
    return max(n, key=n.get)


def stable_seed(*parts) -> int:
    """A 63-bit seed from ``parts`` (``hashlib``, not ``hash()``: the same in every process)."""
    return int(hashlib.sha1("/".join(map(str, parts)).encode()).hexdigest()[:16], 16) % (2**63)


def attempt_seed(seed: int, attempt: int) -> int:
    """Attempt 0 keeps the item's seed (no regeneration = the same audio); attempt k > 0 draws a new one."""
    return seed if attempt == 0 else stable_seed(seed, "retry", attempt)


def _rank(r: dict) -> tuple:
    return (r.get("pace_ok") is False, r["err"])


def synthesize_checked(items: list[dict], synth, check, max_err, max_attempts: int) -> list[dict]:
    """The regeneration loop. ``synth(items) -> wavs``; ``check(wavs, items) -> [{"err", "pace_ok"?, ...}]``;
    ``max_err(item)`` the item's threshold. Every item is synthesized; the ones above their threshold or with
    ``pace_ok`` False again with a new seed, up to ``max_attempts`` in all (not one whose check says ``"exempt"``: too
    short to score reliably; its pace still counts). Per item: the best attempt (in pace first, then the lowest error,
    the earliest on a tie) as ``{"wav", "err", ... check's keys, "seed", "attempts"}`` (``attempts``: how many were
    made)."""
    assert max_attempts >= 1
    best: list[dict | None] = [None] * len(items)
    todo = list(range(len(items)))
    for attempt in range(max_attempts):
        if not todo:
            break
        batch = [{**items[i], "seed": attempt_seed(items[i]["seed"], attempt)} for i in todo]
        wavs = synth(batch)
        res = check(wavs, batch)
        assert len(wavs) == len(res) == len(batch)
        nxt = []
        for i, it, w, r in zip(todo, batch, wavs, res):
            if best[i] is None or _rank(r) < _rank(best[i]):
                best[i] = {**r, "wav": w, "seed": it["seed"]}
            best[i]["attempts"] = attempt + 1
            if (r["err"] > max_err(items[i]) and not r.get("exempt")) or r.get("pace_ok") is False:
                nxt.append(i)
        todo = nxt
    return best


def edge_trim(wav, sr: int, *, lead: bool, trail: bool, db: float = -40.0, margin: float = 0.05) -> tuple[int, int]:
    """``(a, b)``: the sample range of ``wav`` kept when its leading / trailing silence is cut (quieter than ``db``
    below the peak), ``margin`` seconds kept on each cut side."""
    import numpy as np

    n = len(wav)
    if n == 0 or not (lead or trail):
        return 0, n
    peak = float(np.max(np.abs(wav)))
    if peak <= 0:
        return 0, n
    loud = np.flatnonzero(np.abs(wav) > peak * 10 ** (db / 20))
    m = int(margin * sr)
    a = max(0, int(loud[0]) - m) if lead else 0
    b = min(n, int(loud[-1]) + 1 + m) if trail else n
    return (a, b) if b > a else (0, n)


def stats_add(raw: dict, lang: str, err: float, attempts: int, audio_s: float, thr: float,
              pace_ok: bool | None = None) -> None:  # fmt: skip
    """Count one final segment into ``raw`` (``{"dialogues", "time": {...}, "lang": {lang: sums}}``)."""
    d = raw.setdefault("lang", {}).setdefault(
        lang, {"segments": 0, "audio_s": 0.0, "err_sum": 0.0, "regenerated": 0, "above_max_err": 0, "attempts": 0,
               "off_pace": 0}  # fmt: skip
    )
    d["off_pace"] = d.get("off_pace", 0) + (pace_ok is False)
    d["segments"] += 1
    d["audio_s"] += audio_s
    d["err_sum"] += err
    d["regenerated"] += attempts > 1
    d["above_max_err"] += err > thr
    d["attempts"] += attempts


def stats_merge(raws: list[dict]) -> dict:
    """The sum of several workers' ``raw`` stats."""
    out = {"dialogues": 0, "time": {}, "lang": {}}
    for r in raws:
        out["dialogues"] += r.get("dialogues", 0)
        for k, v in r.get("time", {}).items():
            out["time"][k] = max(out["time"].get(k, 0.0), v)  # workers run in parallel: the slowest
        for lg, d in r.get("lang", {}).items():
            o = out["lang"].setdefault(lg, dict.fromkeys(d, 0))
            for k, v in d.items():
                o[k] += v
    return out


def stats_summary(raw: dict) -> dict:
    """Per language: segments, audio hours, mean error, share regenerated, share still above ``max_err`` after the
    last attempt, syntheses per segment; plus dialogues and wall times."""
    out = {"dialogues": raw.get("dialogues", 0), "time": {k: round(v, 1) for k, v in raw.get("time", {}).items()}}
    out["lang"] = {}
    for lg, d in sorted(raw.get("lang", {}).items()):
        n = max(d["segments"], 1)
        out["lang"][lg] = {
            "segments": d["segments"],
            "audio_h": round(d["audio_s"] / 3600, 3),
            "mean_err": round(d["err_sum"] / n, 4),
            "share_regenerated": round(d["regenerated"] / n, 4),
            "share_above_max_err": round(d["above_max_err"] / n, 4),
            "syntheses_per_segment": round(d["attempts"] / n, 3),
            "share_off_pace": round(d.get("off_pace", 0) / n, 4),
        }
    return out


#: The pause before a turn, by who speaks next: ``[mean, std, lo, hi]`` seconds of a normal truncated to [lo, hi]
#: (negative = overlap with the turn before). Users take their time or jump in (high variance); the assistant answers
#: with a steady short latency. ``*`` = any other speaker.
DEFAULT_GAPS = {"user": [0.4, 0.6, -0.3, 2.0], "assistant": [0.35, 0.08, 0.2, 0.55], "*": [0.2, 0.4, -0.3, 0.6]}


def draw_gap(rng, spec) -> float:
    """One pause from ``spec`` = ``[mean, std, lo, hi]`` (resampled until inside [lo, hi])."""
    mean, std, lo, hi = spec
    assert lo <= hi
    for _ in range(1000):
        v = rng.gauss(mean, std)
        if lo <= v <= hi:
            return v
    return min(max(mean, lo), hi)


#: silence kept on each side of a boundary between segments of one turn (seconds)
INNER_MARGIN = 0.02
#: the pause between two segments of one turn, by how the first one ends (seconds; ``segment_gap`` is the floor). A
#: language switch inside a phrase ("Das heißt <en>to commute</en>") gets almost none; one after a sentence or a clause
#: a speaker's pause (trimming every boundary to INNER_MARGIN made sentence ends run into the next sentence, round 3).
BOUNDARY_PAUSE = {"sentence": 0.3, "clause": 0.12}
_END_SENTENCE = re.compile(r"[.!?…。！？]['\"’”»）)]*\s*$")
_END_CLAUSE = re.compile(r"[,;:—–，、；：]['\"’”»）)]*\s*$")


#: silences inside one TTS clip longer than MAX_INNER_PAUSE are shortened to CAPPED_PAUSE (seconds): Chatterbox
#: sometimes pauses 1-2.5 s at a sentence end inside a clip (round 4, designed German voices: p99 1.3 s, max 2.5 s)
MAX_INNER_PAUSE = 0.45
CAPPED_PAUSE = 0.35


def cap_pauses(wav, sr: int, max_s: float = MAX_INNER_PAUSE, to_s: float = CAPPED_PAUSE, db: float = -40.0):
    """``(wav, cuts)``: every inner silence of ``wav`` (10 ms frames quieter than ``db`` below the peak, not touching
    either edge) longer than ``max_s`` shortened to ``to_s`` by removing its middle; ``cuts`` = the removed spans as
    ``(start, length)`` seconds on the input's timeline (:func:`shift_time` maps times through them)."""
    import numpy as np

    hop = max(1, int(0.01 * sr))
    n = len(wav) // hop
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if n < 3 or peak <= 0:
        return wav, []
    sil = np.abs(wav[: n * hop]).reshape(n, hop).max(1) < peak * 10 ** (db / 20)
    keep, cuts, last, k = [], [], 0, 0
    half = int(to_s * sr / 2)
    while k < n:
        if not sil[k]:
            k += 1
            continue
        j = k
        while j < n and sil[j]:
            j += 1
        if k > 0 and j < n and (j - k) * hop > max_s * sr:
            a, b = k * hop + half, j * hop - half
            keep.append(wav[last:a])
            cuts.append((a / sr, (b - a) / sr))
            last = b
        k = j
    if not cuts:
        return wav, []
    keep.append(wav[last:])
    return np.concatenate(keep), cuts


def shift_time(t: float, cuts: list) -> float:
    """Time ``t`` of a clip after :func:`cap_pauses` removed ``cuts`` (a time inside a removed span lands on its
    start)."""
    out = t
    for a, n in cuts:
        if t >= a + n:
            out -= n
        elif t > a:
            out -= t - a
    return out


def boundary_pause(text: str, segment_gap: float) -> float:
    """The pause after a segment whose text is ``text`` when another segment of the same turn follows."""
    if _END_SENTENCE.search(text):
        return max(segment_gap, BOUNDARY_PAUSE["sentence"])
    if _END_CLAUSE.search(text):
        return max(segment_gap, BOUNDARY_PAUSE["clause"])
    return segment_gap

#: an interrupted turn's audio ends this long after the end of its last word before the cut (then a 40 ms fade)
CUT_TAIL = 0.12
#: how long the interrupting turn overlaps the interrupted one: uniform in this range (seconds); the speaker stops
#: this long after the other started
DEFAULT_CUT_OVERLAP = [0.25, 0.6]

#: every turn's speech level, dB of full scale (RMS over its sounding samples); peaks kept under -1 dBFS
DEFAULT_LOUDNESS_DBFS = -23.0  # Turbo's level (it normalises the voice prompt): median -23 in the pilot


def level_to(wav, target_dbfs: float | None, *, db: float = -40.0, peak_dbfs: float = -1.0):
    """``wav`` scaled so the RMS of its sounding samples (louder than ``db`` below its peak) is ``target_dbfs``, the
    gain capped so no sample passes ``peak_dbfs``. ``None``: unchanged. Chatterbox's output level follows the voice
    prompt's (Multilingual V3 does not normalise it): pilot 2026-11-17, Multilingual turns -49 to -19 dBFS (p5-p95),
    turns of one dialogue 15.5 dB apart at p90 (Turbo -25 to -21)."""
    import numpy as np

    if target_dbfs is None or len(wav) == 0:
        return wav
    a = np.abs(wav)
    peak = float(a.max())
    if peak <= 0:
        return wav
    loud = wav[a > peak * 10 ** (db / 20)]
    rms = float(np.sqrt(np.mean(loud.astype(np.float64) ** 2)))
    gain = min(10 ** (target_dbfs / 20) / max(rms, 1e-9), 10 ** (peak_dbfs / 20) / peak)
    return (wav * gain).astype(np.float32)


def render(plan: dict, results: list[dict], sr: int, segment_gap: float,
           loudness_dbfs: float | None = DEFAULT_LOUDNESS_DBFS) -> tuple[dict, list, list, list]:  # fmt: skip
    """Speaker tracks, turn rows, segment rows and word rows of one planned dialogue (``chatterbox_v2_inference.
    plan_dialogue``: ``turns``, ``segs``, ``items``, ``gaps``) from its segment results (``synthesize_checked``'s, with
    ``words`` in seconds from each segment's start). Every segment's leading and trailing silence is trimmed (so a
    turn starts with speech and the pause before it is exactly the planned gap: Chatterbox Multilingual leads with up to
    1 s of silence, measured 2026-11-17), long silences inside a clip shortened (:func:`cap_pauses`), a turn's segments joined with :func:`boundary_pause` between, each turn brought to
    ``loudness_dbfs`` (:func:`level_to`); turns follow ``chatterbox_inference``'s timeline (``pre_silence``, then the turn, then its
    gap; a negative gap overlaps the next turn, never before this turn's start). Times in the rows are seconds on the
    dialogue timeline; ``turns.text`` is the plain text (:func:`turn_plain_text`)."""
    import numpy as np

    by_turn: dict[int, list] = {}
    for it, r in zip(plan["items"], results):
        by_turn.setdefault(it["turn"], []).append((it, r))
    pieces, turn_rows, seg_rows, word_rows = [], [], [], []
    t_samples = 0
    for ti, turn in enumerate(plan["turns"]):
        if turn.get("pre_silence"):
            assert turn["pre_silence"] >= 0, "pre_silence must be non-negative"
            t_samples += int(sr * turn["pre_silence"])
        segs = by_turn[ti]
        start, pos, chunks = t_samples, t_samples, []
        for k, (it, r) in enumerate(segs):
            # turn edges keep 50 ms; a language switch inside the turn only INNER_MARGIN (the pilot's 50 ms + 80 ms gap +
            # 50 ms sounded like a pause between the languages)
            a, _ = edge_trim(r["wav"], sr, lead=True, trail=False, margin=0.05 if k == 0 else INNER_MARGIN)
            _, b = edge_trim(r["wav"], sr, lead=False, trail=True, margin=0.05 if k == len(segs) - 1 else INNER_MARGIN)
            if k > 0:
                g = int(boundary_pause(segs[k - 1][0]["text"], segment_gap) * sr)
                chunks.append(np.zeros(g, np.float32))
                pos += g
            clip, cuts = cap_pauses(r["wav"][a:b], sr)
            chunks.append(clip)
            s0 = pos / sr
            seg_rows.append({"turn": ti, "lang": it["lang"], "text": it["text"], "start": s0, "end": s0 + len(clip) / sr,
                             "err": r["err"], "wer": r.get("wer", r["err"]), "cer": r.get("cer", r["err"]),
                             "attempts": r["attempts"], "asr_text": r["asr_text"],
                             "rate": float(r.get("rate", 0.0))})  # fmt: skip
            for w in r["words"]:
                ws, we = shift_time(w["start"] - a / sr, cuts), shift_time(w["end"] - a / sr, cuts)
                if we <= 0 or ws >= len(clip) / sr:
                    continue  # inside a trimmed edge (should not happen: trimmed parts are silent)
                word_rows.append({"turn": ti, "segment": it["segment"], "speaker": turn["speaker"], "lang": it["lang"],
                                  "text": w["text"], "start": s0 + max(ws, 0.0),
                                  "end": s0 + min(we, len(clip) / sr)})  # fmt: skip
            pos += len(clip)
        wav = level_to(np.concatenate(chunks), loudness_dbfs)
        n_keep = turn_cut_units(turn, plan.get("lang") or "en")
        tw = [w for w in word_rows if w["turn"] == ti]
        if n_keep is not None and 0 < n_keep < len(tw):
            t_cut = tw[n_keep - 1]["end"] + CUT_TAIL  # dialogue time
            n = min(len(wav), max(1, int(round(t_cut * sr)) - start))
            wav = wav[:n].copy()
            f = min(n, int(0.04 * sr))
            wav[n - f :] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
            drop = {id(w) for w in tw[n_keep:]}
            word_rows = [w for w in word_rows if id(w) not in drop]
            seg_rows = [x for x in seg_rows if x["turn"] != ti or x["start"] < t_cut]
            for x in seg_rows:
                if x["turn"] == ti:
                    x["end"] = min(x["end"], t_cut)
        pieces.append((turn["speaker"], start, wav))
        turn_rows.append({"speaker": turn["speaker"], "start_time": start / sr, "end_time": (start + len(wav)) / sr,
                          "text": turn_plain_text(turn), "lang": main_lang(plan["segs"][ti]),
                          "interrupted": n_keep is not None})  # fmt: skip
        t_samples = start + len(wav)
        gap = plan["gaps"][ti]
        delta = int(sr * gap)
        t_samples = t_samples + delta if gap >= 0 else max(start, t_samples + delta)
    end = max(s + len(w) for _, s, w in pieces)
    tracks = {sp: np.zeros(end, np.float32) for sp in dict.fromkeys(t["speaker"] for t in plan["turns"])}
    for sp, s, w in pieces:
        tracks[sp][s : s + len(w)] += w
    return tracks, turn_rows, seg_rows, word_rows
