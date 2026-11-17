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


def turn_segments(turn: dict, lang: str) -> list[dict]:
    """A dialogue turn's segments: its ``segments`` if given, else its ``text`` parsed in ``turn["lang"]`` or ``lang``."""
    if turn.get("segments"):
        return [{"lang": s["lang"], "text": s["text"].strip()} for s in turn["segments"] if _has_content(s["text"])]
    text = turn["text"]
    if isinstance(text, list):  # some writers emit a one-element list
        text = " ".join(str(t) for t in text)
    return parse_segments(text, turn.get("lang") or lang)


def turn_plain_text(turn: dict) -> str:
    """A turn's text as a model's text stream sees it: no language tags."""
    if turn.get("segments"):
        return join_segments([s["text"] for s in turn["segments"]])
    text = turn["text"]
    return plain_text(" ".join(str(t) for t in text) if isinstance(text, list) else text)


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


def synthesize_checked(items: list[dict], synth, check, max_err, max_attempts: int) -> list[dict]:
    """The regeneration loop. ``synth(items) -> wavs``; ``check(wavs, items) -> [{"err", ...}]``; ``max_err(item)`` the
    item's threshold. Every item is synthesized; the ones above their threshold again with a new seed, up to
    ``max_attempts`` in all (not one whose check says ``"exempt"``: too short to score reliably). Per item: the attempt with the lowest error (the earliest on a tie) as ``{"wav", "err", ...
    check's keys, "seed", "attempts"}`` (``attempts``: how many were made)."""
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
            if best[i] is None or r["err"] < best[i]["err"]:
                best[i] = {**r, "wav": w, "seed": it["seed"]}
            best[i]["attempts"] = attempt + 1
            if r["err"] > max_err(items[i]) and not r.get("exempt"):
                nxt.append(i)
        todo = nxt
    return best


def edge_trim(wav, sr: int, *, lead: bool, trail: bool, db: float = -40.0, margin: float = 0.05) -> tuple[int, int]:
    """``(a, b)``: the sample range of ``wav`` kept when its leading / trailing silence is cut (a turn's INNER segment
    boundaries; quieter than ``db`` below the peak), ``margin`` seconds kept on each cut side."""
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


def stats_add(raw: dict, lang: str, err: float, attempts: int, audio_s: float, thr: float) -> None:
    """Count one final segment into ``raw`` (``{"dialogues", "time": {...}, "lang": {lang: sums}}``)."""
    d = raw.setdefault("lang", {}).setdefault(
        lang, {"segments": 0, "audio_s": 0.0, "err_sum": 0.0, "regenerated": 0, "above_max_err": 0, "attempts": 0}
    )
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
        }
    return out


def render(plan: dict, results: list[dict], sr: int, segment_gap: float) -> tuple[dict, list, list, list]:
    """Speaker tracks, turn rows, segment rows and word rows of one planned dialogue (``chatterbox_v2_inference.
    plan_dialogue``: ``turns``, ``segs``, ``items``, ``gaps``) from its segment results (``synthesize_checked``'s, with
    ``words`` in seconds from each segment's start). A turn's segments are joined with their inner silence trimmed and
    ``segment_gap`` between; turns follow ``chatterbox_inference``'s timeline (``pre_silence``, then the turn, then its
    gap; a negative gap overlaps the next turn, never before this turn's start). Times in the rows are seconds on the
    dialogue timeline; ``turns.text`` is the plain text (:func:`turn_plain_text`)."""
    import numpy as np

    by_turn: dict[int, list] = {}
    for it, r in zip(plan["items"], results):
        by_turn.setdefault(it["turn"], []).append((it, r))
    pieces, turn_rows, seg_rows, word_rows = [], [], [], []
    t_samples = 0
    gap_n = int(segment_gap * sr)
    for ti, turn in enumerate(plan["turns"]):
        if turn.get("pre_silence"):
            assert turn["pre_silence"] >= 0, "pre_silence must be non-negative"
            t_samples += int(sr * turn["pre_silence"])
        segs = by_turn[ti]
        start, pos, chunks = t_samples, t_samples, []
        for k, (it, r) in enumerate(segs):
            a, b = edge_trim(r["wav"], sr, lead=k > 0, trail=k < len(segs) - 1)
            if k > 0:
                chunks.append(np.zeros(gap_n, np.float32))
                pos += gap_n
            chunks.append(r["wav"][a:b])
            s0 = pos / sr
            seg_rows.append({"turn": ti, "lang": it["lang"], "text": it["text"], "start": s0, "end": s0 + (b - a) / sr,
                             "err": r["err"], "wer": r.get("wer", r["err"]), "cer": r.get("cer", r["err"]),
                             "attempts": r["attempts"], "asr_text": r["asr_text"]})  # fmt: skip
            for w in r["words"]:
                ws, we = w["start"] - a / sr, w["end"] - a / sr
                if we <= 0 or ws >= (b - a) / sr:
                    continue  # inside a trimmed edge (should not happen: trimmed parts are silent)
                word_rows.append({"turn": ti, "segment": it["segment"], "speaker": turn["speaker"], "lang": it["lang"],
                                  "text": w["text"], "start": s0 + max(ws, 0.0),
                                  "end": s0 + min(we, (b - a) / sr)})  # fmt: skip
            pos += b - a
        wav = np.concatenate(chunks)
        pieces.append((turn["speaker"], start, wav))
        turn_rows.append({"speaker": turn["speaker"], "start_time": start / sr, "end_time": (start + len(wav)) / sr,
                          "text": turn_plain_text(turn), "lang": main_lang(plan["segs"][ti])})  # fmt: skip
        t_samples = start + len(wav)
        gap = plan["gaps"][ti]
        delta = int(sr * gap)
        t_samples = t_samples + delta if gap >= 0 else max(start, t_samples + delta)
    end = max(s + len(w) for _, s, w in pieces)
    tracks = {sp: np.zeros(end, np.float32) for sp in dict.fromkeys(t["speaker"] for t in plan["turns"])}
    for sp, s, w in pieces:
        tracks[sp][s : s + len(w)] += w
    return tracks, turn_rows, seg_rows, word_rows
