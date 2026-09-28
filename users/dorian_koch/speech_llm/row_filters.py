"""Row keep-lists and per-row loss weights for PersonaPlex window corpora (2026-09-28).

The pattern of the episode-edge filter (``persona_prompts.EpisodeEdgeRows``), generalised: a cheap CPU job reads an
EXISTING window corpus (+ its window context / dialogue slices) and writes

  * a keep-list ``rows.npy`` -- int64 table-row indices, ascending (the launcher's ``train_rows``,
    ``train_data_common.load_row_subset``); or
  * per-row loss weights ``weights.npy`` -- float32, one per TABLE row (the launcher's ``train_row_weights``,
    ``train_data_common.load_row_weights``).

No corpus rebuild, no re-prompting, nothing copied. Jobs:

  MonologueRows        rows whose longest single-speaker stretch inside the window is <= ``max_stretch_sec``
                       (:func:`longest_stretch`: ``podcast_ingest.select_windows``' own stretch definition), plus
                       the per-row stretch (``stretch.npy``) for :class:`ThresholdRowWeights`.
  LexiconRows          rows none of whose TRAINABLE sides says a word of a fixed lexicon (:data:`SAFETY_LEXICONS`).
  IntersectRows        the rows every given keep-list keeps: the composition (e.g. edge AND monologue AND lexicon).
                       EpisodeEdgeRows is untouched; an arm with only the edge filter keeps its keep-list and hash.
  ThresholdRowWeights  ``weight`` on rows whose value (e.g. ``MonologueRows.out_stretch``) exceeds ``above``, else 1.
  SafetyLabelSample    GPU (vLLM). LLM content labels per speaker on a seeded sample of windows (a MEASUREMENT: how
                       much sexual / profane / violent / hateful / drug content there is, and how well the lexicon
                       finds it). Not a filter.

"Trainable side": a PersonaPlex row is read with the stored assistant channel as the assistant, or -- when
``swap_channels`` is on and the row's ``other_ok`` -- the other channel (1/2 each read). Both are trained as the
assistant, so a content filter on "the assistant side" must look at both (``sides="trainable"``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from sisyphus import Job, Task, tk

from .common import map_concurrent, vllm_gpu_mem_gb, vllm_server

#: Caps every MonologueRows summary tabulates (rows / hours kept at each), whatever its own cap.
STRETCH_CAPS = (10.0, 15.0, 20.0, 25.0, 30.0)


# ---------------------------------------------------------------------------------------------
# keep-list / weight file helpers
# ---------------------------------------------------------------------------------------------
def load_keep_list(path) -> "np.ndarray":  # noqa: F821
    """A keep-list as ``train_data_common.load_row_subset`` reads it: 1-D int, ascending, unique, non-negative."""
    import numpy as np

    rows = np.load(str(path.get_path() if hasattr(path, "get_path") else path))
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"{path}: a keep-list is a 1-D integer array, got {rows.dtype} {rows.shape}")
    rows = rows.astype(np.int64)
    if rows.size and (rows[0] < 0 or np.any(np.diff(rows) <= 0)):
        raise ValueError(f"{path}: a keep-list is ascending, unique and non-negative")
    return rows


def _rows_hours(idx, dur) -> dict:
    return {"rows": int(len(idx)), "hours": round(float(sum(float(dur[i]) for i in idx)) / 3600.0, 2)}


# ---------------------------------------------------------------------------------------------
# monologue: the longest single-speaker stretch inside a window
# ---------------------------------------------------------------------------------------------
def merged_runs(turns: list[dict]) -> list[list]:
    """``select_windows``' stretch runs: ALL diarized turns (any speaker, any length) sorted by start, consecutive
    same-speaker turns merged (end = the max end). ``[[speaker, start, end], ...]``."""
    ts = sorted(((str(t["speaker"]), float(t["start"]), float(t["end"])) for t in turns), key=lambda t: t[1])
    runs = []
    for t in ts:
        if runs and runs[-1][0] == t[0]:
            runs[-1][2] = max(runs[-1][2], t[2])
        else:
            runs.append(list(t))
    return runs


def longest_stretch(runs: list[list], a: float, b: float) -> tuple:
    """``(seconds, speaker)`` of the longest run of :func:`merged_runs` clipped to the window ``[a, b]`` -- exactly the
    quantity ``select_windows`` holds to ``max_stretch_sec`` (``np.clip(min(end, b) - max(start, a), 0).max()``);
    speaker None when no run overlaps the window. Ties: the earliest run."""
    best, who = 0.0, None
    for spk, s, e in runs:
        x = min(e, b) - max(s, a)
        if x > best:
            best, who = x, spk
    return best, who


def _slice_parts(d) -> list[str]:
    root = d.get_path() if hasattr(d, "get_path") else str(d)
    pd = os.path.join(root, "parts")
    return sorted(os.path.join(pd, n) for n in os.listdir(pd) if not n.endswith(".tmp"))


def window_stretches(context_path: str, slice_dirs: list, ids=None) -> dict:
    """``{window id: (stretch sec, holder)}`` for the windows of a ``PodcastWindowContext`` (``ids``: only these).
    The window is the context row's ``[t0, t1]`` (the frame-aligned bounds the training row was cut to, relative to
    its dialogue row), the turns are the dialogue slice's ``turns_json``. ``holder``: ``"assistant"`` when the
    stretch's diarized speaker is the row's ``assistant_label`` (the STORED assistant channel), ``"other"`` when it
    is its ``user_label``, ``"third"`` otherwise (a third speaker, or a side without a label), None when no turn
    overlaps the window. Raises on a window whose dialogue row is not in ``slice_dirs``."""
    from datasets import load_from_disk

    full = load_from_disk(str(context_path))
    cols = ["id", "item_id", "t0", "t1", "assistant_label"] + (
        ["user_label"] if "user_label" in full.column_names else []
    )
    rows = [(r, r.get("user_label")) for r in full.select_columns(cols) if ids is None or r["id"] in ids]
    items = {r["item_id"] for r, _ in rows}
    turns = {}
    for d in slice_dirs:
        for p in _slice_parts(d):
            part = load_from_disk(p).select_columns(["item_id", "turns_json"])
            for it, tj in zip(part["item_id"], part["turns_json"]):
                if it in items:
                    turns[it] = merged_runs(json.loads(tj))
    missing = items - set(turns)
    if missing:
        raise RuntimeError(
            f"{len(missing)} dialogue rows of {context_path} not in its slices (e.g. {sorted(missing)[:3]})"
        )
    out = {}
    for r, ul in rows:
        sec, spk = longest_stretch(turns[r["item_id"]], float(r["t0"]), float(r["t1"]))
        if spk is None:
            holder = None
        elif spk == r["assistant_label"]:
            holder = "assistant"
        elif ul is not None and spk == ul:
            holder = "other"
        else:
            holder = "third"
        out[r["id"]] = (float(sec), holder)
    return out


def monologue_rows_summary(
    data_path: str, context_path: str, slice_dirs: list, max_stretch_sec: float, caps=STRETCH_CAPS
) -> tuple:
    """:class:`MonologueRows`' worker: ``(summary, keep, stretch)``. ``stretch``: float32 per TABLE row of
    ``data_path``; ``keep``: the rows with ``stretch <= max_stretch_sec``, ascending. Also the measurement behind the
    cap (distribution, rows / hours kept at every cap of ``caps``, and who holds the over-cap stretches)."""
    import numpy as np
    from datasets import load_from_disk

    meta = load_from_disk(str(data_path)).select_columns(["id", "duration"])
    ids, dur = meta["id"], np.asarray(meta["duration"], dtype=np.float64)
    st = window_stretches(context_path, slice_dirs, ids=set(ids))
    lost = [i for i in ids if i not in st]
    if lost:
        raise RuntimeError(f"{len(lost)} rows of {data_path} not in the window context (e.g. {lost[:3]})")
    stretch = np.asarray([st[i][0] for i in ids], dtype=np.float32)
    holder = [st[i][1] for i in ids]
    keep = np.flatnonzero(stretch <= float(max_stretch_sec)).astype(np.int64)
    tot_h = float(dur.sum()) / 3600.0
    per_cap = {}
    for c in sorted({float(x) for x in caps} | {float(max_stretch_sec)}):
        ok = stretch <= c
        over = [h for h, o in zip(holder, ok) if not o]
        per_cap[f"{c:g}"] = {
            "rows_kept": int(ok.sum()),
            "rows_kept_share": round(float(ok.mean()), 4),
            "hours_kept": round(float(dur[ok].sum()) / 3600.0, 2),
            "hours_kept_share": round(float(dur[ok].sum()) / 3600.0 / tot_h, 4),
            "dropped_by_holder": {k: over.count(k) for k in ("assistant", "other", "third", None) if over.count(k)},
        }
    qs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    edges = list(range(0, 35, 5))
    hist = np.histogram(stretch, bins=edges + [float("inf")])[0]
    summary = {
        "max_stretch_sec": float(max_stretch_sec),
        "rule": "keep a row iff the longest single-speaker stretch inside its window <= max_stretch_sec "
        "(select_windows' stretch: all diarized turns, same-speaker neighbours merged, clipped to [t0, t1])",
        "rows": len(ids),
        "hours": round(tot_h, 2),
        "kept": _rows_hours(keep, dur),
        "stretch_quantiles": {f"p{int(q * 100)}": round(float(np.quantile(stretch, q)), 2) for q in qs},
        "stretch_mean": round(float(stretch.mean()), 2),
        "stretch_max": round(float(stretch.max()), 2),
        "stretch_hist_5s": {
            f"{a}-{b}" if b != float("inf") else f">={a}": int(n)
            for a, b, n in zip(edges, edges[1:] + [float("inf")], hist)
        },
        "holder_of_longest": {str(k): holder.count(k) for k in ("assistant", "other", "third", None)},
        "caps": per_cap,
    }
    return summary, keep, stretch


class MonologueRows(Job):
    """The rows of a window corpus whose longest single-speaker stretch inside the window is at most
    ``max_stretch_sec`` (user, 2026-09-28: "I feel like longest monologue shouldnt be longer than 20, maybe 15s").
    The windows were cut with ``select_windows(max_stretch_sec=30)`` (``w20_60_t3_s30``), so every row is <= 30 s
    already; this tightens the same quantity (:func:`longest_stretch`) without re-cutting. Measured on p4's corpus
    before any default (rules.md / backlog "monologue cap").

    ``data``: the training corpus (rows carry ``id`` + ``duration``); ``window_context`` + ``slice_dirs``: the
    ``PodcastWindowContext`` its ids come from and that context's dialogue slices (the same pair the episode-edge
    filter uses, ``track_persona.PPX_EDGE_SOURCES``).

    Outputs: ``out_rows`` (``rows.npy``, the keep-list), ``out_stretch`` (``stretch.npy``: float32 seconds per TABLE
    row, for :class:`ThresholdRowWeights`) and ``out_summary`` (distribution, rows / hours kept at every
    :data:`STRETCH_CAPS` cap, holder of the dropped stretches)."""

    def __init__(self, *, data: tk.Path, window_context: tk.Path, slice_dirs: list, max_stretch_sec: float = 20.0):
        assert 0.0 < float(max_stretch_sec), max_stretch_sec
        self.data = data
        self.window_context = window_context
        self.slice_dirs = list(slice_dirs)
        self.max_stretch_sec = float(max_stretch_sec)
        self.out_rows = self.output_path("rows.npy")
        self.out_stretch = self.output_path("stretch.npy")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        import numpy as np

        summary, keep, stretch = monologue_rows_summary(
            self.data.get_path(), self.window_context.get_path(), self.slice_dirs, self.max_stretch_sec
        )
        np.save(self.out_rows.get_path(), keep)
        np.save(self.out_stretch.get_path(), stretch)
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[monologue] <= {self.max_stretch_sec:g} s: {summary['kept']} of {summary['rows']} rows", flush=True)


# ---------------------------------------------------------------------------------------------
# lexicon: a fixed word list per content category
# ---------------------------------------------------------------------------------------------
#: Fixed lexicons (regexes over lower-cased words, apostrophes kept, other punctuation dropped; whole words only).
#: Hashed by digest into every job that uses one: an edited lexicon is a new name. ``explicit_v1`` (2026-09-28) is
#: deliberately small and HIGH-PRECISION: ambiguous words are left out ("dick" = a name, "ass", "damn", "hell",
#: "naked", "cum", "cock"), so it under-counts; the LLM sample (SafetyLabelSample) measures by how much.
SAFETY_LEXICONS = {
    "explicit_v1": {
        "profanity": [
            r"(?:mother)?fuck\w*",
            r"fuckin'?",
            r"(?:bull|horse|dip|dog)?shit(?:s|ty|ting|ted|head|heads|show)?",
            r"bitch(?:es|y|ing|in'?)?",
            r"cunts?",
            r"assholes?",
            r"dickheads?",
            r"cocksuckers?",
            r"goddamn(?:ed|it)?",
        ],
        "sexual": [
            r"sex",
            r"sexy",
            r"sexual(?:ly)?",
            r"porn\w*",
            r"orgasm\w*",
            r"masturbat\w*",
            r"blow ?jobs?",
            r"hand ?jobs?",
            r"jerk(?:ing|ed|s)? off",
            r"horny",
            r"penis(?:es)?",
            r"vaginas?",
            r"puss(?:y|ies)",
            r"boobs?",
            r"tits",
            r"titties",
            r"dildos?",
            r"threesomes?",
            r"anal",
            r"erections?",
            r"boners?",
            r"nudes?",
            r"hookers?",
            r"prostitut\w*",
            r"strippers?",
            r"fetish(?:es)?",
            r"kinky",
            r"clit(?:oris)?",
            r"semen",
        ],
        "slur": [
            r"nigg(?:er|a|ers|as|az)",
            r"fag(?:s|got|gots)?",
            r"retard(?:s|ed)?",
            r"trann(?:y|ies)",
            r"kikes?",
            r"spics?",
            r"chinks?",
        ],
    },
}


def lexicon_digest(name: str) -> str:
    return hashlib.sha256(json.dumps(SAFETY_LEXICONS[name], sort_keys=True).encode()).hexdigest()[:16]


def norm_words(text: str) -> str:
    """Lower-case, curly apostrophes straightened, every character but ``a-z0-9'`` a space, single-spaced."""
    t = str(text or "").lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t).split())


def lexicon_patterns(name: str) -> dict:
    """``{category: compiled regex}``: any of the category's words as a whole word (bounded by space / text end)."""
    return {
        cat: re.compile(r"(?:(?<= )|^)(?:" + "|".join(ws) + r")(?=$| )") for cat, ws in SAFETY_LEXICONS[name].items()
    }


def lexicon_hits(text: str, pats: dict) -> dict:
    """``{category: number of matches}`` in ``text`` (normalised by :func:`norm_words`)."""
    t = norm_words(text)
    return {cat: len(p.findall(t)) for cat, p in pats.items()}


def _side_text(alignments) -> str:
    return " ".join(str(w["text"]) for w in (alignments or []))


def lexicon_rows_summary(data_path: str, lexicon: str, categories, sides: str) -> tuple:
    """:class:`LexiconRows`' worker: ``(summary, keep, flags)``. A row is DROPPED when a side named by ``sides``
    says a word of one of ``categories``: ``"trainable"`` = the stored assistant always + the other side when
    ``other_ok`` (both are trained as the assistant under swap_channels), ``"assistant"`` = the stored assistant
    only, ``"either"`` = both regardless of ``other_ok``. ``flags``: ``{row index: {side: {category: hits}}}`` of
    every row with any hit. The summary counts per side and category, per window and per word, over ALL rows."""
    import numpy as np
    from datasets import load_from_disk

    assert sides in ("trainable", "assistant", "either"), sides
    cats = tuple(categories)
    pats = lexicon_patterns(lexicon)
    assert set(cats) <= set(pats), (cats, sorted(pats))
    ds = load_from_disk(str(data_path))
    cols = ["id", "duration", "alignments"] + [c for c in ("alignments_other", "other_ok") if c in ds.column_names]
    ds = ds.select_columns(cols)
    has_other = "alignments_other" in cols
    keep, flags = [], {}
    dur = np.asarray(ds["duration"], dtype=np.float64)
    stat = {
        s: {"windows": 0, "words": 0, **{c: {"windows": 0, "hits": 0} for c in pats}} for s in ("assistant", "other")
    }
    oks = []
    for i, r in enumerate(ds):
        other_ok = bool(r.get("other_ok", False)) if has_other else False
        oks.append(other_ok)
        texts = {"assistant": _side_text(r["alignments"])}
        if has_other:
            texts["other"] = _side_text(r["alignments_other"])
        hits = {}
        for s, t in texts.items():
            h = lexicon_hits(t, pats)
            st = stat[s]
            st["windows"] += 1
            st["words"] += len(norm_words(t).split())
            for c, n in h.items():
                st[c]["hits"] += n
                st[c]["windows"] += n > 0
            if any(h.values()):
                hits[s] = h
        if hits:
            flags[i] = hits
        judged = {"assistant"} | (
            {"other"} if (sides == "either" and has_other) or (sides == "trainable" and other_ok) else set()
        )
        if not any(hits.get(s, {}).get(c, 0) for s in judged for c in cats):
            keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    dropped = sorted(set(range(len(dur))) - set(keep.tolist()))
    per_side = {}
    for s, st in stat.items():
        if not st["windows"]:
            continue
        per_side[s] = {
            "windows": st["windows"],
            "words": st["words"],
            **{
                c: {
                    "windows": st[c]["windows"],
                    "window_share": round(st[c]["windows"] / st["windows"], 5),
                    "hits": st[c]["hits"],
                    "per_1k_words": round(1000.0 * st[c]["hits"] / max(st["words"], 1), 4),
                }
                for c in pats
            },
        }
    trainable_any = {
        c: sum(
            1 for i, h in flags.items() if h.get("assistant", {}).get(c, 0) or (h.get("other", {}).get(c, 0) and oks[i])
        )
        for c in pats
    }
    summary = {
        "lexicon": lexicon,
        "lexicon_digest": lexicon_digest(lexicon),
        "categories_dropped": list(cats),
        "sides": sides,
        "rule": "drop a row when a judged side (see sides) says a lexicon word of a dropped category",
        "rows": len(dur),
        "hours": round(float(dur.sum()) / 3600.0, 2),
        "other_ok_rows": int(sum(oks)),
        "kept": _rows_hours(keep, dur),
        "dropped": _rows_hours(dropped, dur),
        "per_side": per_side,
        "rows_with_hit_on_a_trainable_side": trainable_any,
    }
    return summary, keep, flags


class LexiconRows(Job):
    """The rows of a window corpus whose TRAINABLE sides say no word of a fixed lexicon category (user, 2026-09-28:
    "Maybe we should also start filtering out windows that contain 18+ words ... In the self play they use swear words
    or talk about sex"). ``lexicon``: a :data:`SAFETY_LEXICONS` name (its digest is hashed); ``categories``: the
    categories that drop a row (default ``("sexual",)``: profanity is common in JRE and measured, not dropped by
    default); ``sides``: which sides are judged (:func:`lexicon_rows_summary`; default ``"trainable"``).

    Outputs: ``out_rows`` (``rows.npy``), ``out_summary`` (per side and category: windows with a hit, hits per 1k
    words; rows / hours kept and dropped) and ``out_flags`` (json lines ``{"row", "id", "hits"}`` of every row with
    any hit, all categories -- the input for a later loss mask or an audit)."""

    def __init__(
        self,
        *,
        data: tk.Path,
        lexicon: str = "explicit_v1",
        categories: tuple = ("sexual",),
        sides: str = "trainable",
    ):
        assert lexicon in SAFETY_LEXICONS, lexicon
        assert set(categories) <= set(SAFETY_LEXICONS[lexicon]), categories
        assert sides in ("trainable", "assistant", "either"), sides
        self.data = data
        self.lexicon = lexicon
        self.lexicon_digest = lexicon_digest(lexicon)  # hashed: an edited lexicon is a different job
        self.categories = tuple(sorted(categories))
        self.sides = sides
        self.out_rows = self.output_path("rows.npy")
        self.out_summary = self.output_path("summary.json")
        self.out_flags = self.output_path("flags.jsonl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        import numpy as np
        from datasets import load_from_disk

        summary, keep, flags = lexicon_rows_summary(self.data.get_path(), self.lexicon, self.categories, self.sides)
        ids = load_from_disk(self.data.get_path()).select_columns(["id"])["id"]
        np.save(self.out_rows.get_path(), keep)
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        with open(self.out_flags.get_path(), "w") as f:
            for i in sorted(flags):
                f.write(json.dumps({"row": i, "id": ids[i], "hits": flags[i]}) + "\n")
        print(f"[lexicon] kept {summary['kept']} dropped {summary['dropped']}", flush=True)


# ---------------------------------------------------------------------------------------------
# composition and weights
# ---------------------------------------------------------------------------------------------
class IntersectRows(Job):
    """The rows EVERY keep-list in ``rows`` keeps (ascending), e.g. ``[EpisodeEdgeRows.out_rows,
    MonologueRows.out_rows]``: how criteria compose into the one ``train_rows`` a run takes. The inputs are
    unchanged jobs, so an arm using only one of them keeps that job's hash. ``rows`` is hashed in the given order
    (callers pass a fixed order: edge, monologue, lexicon; ``track_persona.ppx_train_rows``).

    Outputs: ``out_rows`` and ``out_summary`` (each input's size, the result, and what each input alone removes from
    the intersection of the others)."""

    def __init__(self, *, rows: list):
        assert len(rows) >= 2, "an intersection of fewer than two keep-lists is the keep-list itself"
        self.rows = list(rows)
        self.out_rows = self.output_path("rows.npy")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        import numpy as np

        lists = [load_keep_list(p) for p in self.rows]
        out = intersect_keep_lists(lists)
        np.save(self.out_rows.get_path(), out)
        summary = {
            "inputs": [{"path": str(p), "rows": int(len(x))} for p, x in zip(self.rows, lists)],
            "kept": int(len(out)),
            # rows only input k drops, given the others: |intersection of the others| - |all|
            "unique_drops": [
                int(len(intersect_keep_lists([y for j, y in enumerate(lists) if j != k])) - len(out))
                if len(lists) > 2
                else int(len(lists[1 - k]) - len(out))
                for k in range(len(lists))
            ],
        }
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[intersect] {summary}", flush=True)


def intersect_keep_lists(lists: list) -> "np.ndarray":  # noqa: F821
    import numpy as np

    out = lists[0]
    for x in lists[1:]:
        out = np.intersect1d(out, x, assume_unique=True)
    if not len(out):
        raise RuntimeError("the keep-lists share no row: a run would train on nothing")
    return out.astype(np.int64)


class ThresholdRowWeights(Job):
    """Per-row loss weights: ``weight`` on every table row whose value in ``values`` (a float ``.npy`` with one entry
    per table row, e.g. ``MonologueRows.out_stretch``) is ABOVE ``above``, 1.0 elsewhere (user, 2026-09-28: "maybe
    we could downweight the loss on the entire row if it contains longer monologues?"). The launcher's
    ``train_row_weights``: every loss target of such a row (text + all audio codebooks) is scaled by the weight in
    the numerator only (``duplex_ce_loss(sample_frame_weight=)``), so the row counts ``weight`` x as much in its
    batch and the other rows are not scaled up.

    Outputs: ``out_weights`` (``weights.npy``, float32 per table row) and ``out_summary``."""

    def __init__(self, *, values: tk.Path, above: float, weight: float):
        assert 0.0 <= float(weight) < 1.0, "a down-weight in [0, 1)"
        self.values = values
        self.above = float(above)
        self.weight = float(weight)
        self.out_weights = self.output_path("weights.npy")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        import numpy as np

        v = np.load(self.values.get_path())
        w = threshold_weights(v, self.above, self.weight)
        np.save(self.out_weights.get_path(), w)
        n = int((w != 1.0).sum())
        summary = {
            "rows": int(len(w)),
            "above": self.above,
            "weight": self.weight,
            "down_weighted": n,
            "share": round(n / len(w), 4),
        }
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[weights] {summary}", flush=True)


def threshold_weights(values, above: float, weight: float) -> "np.ndarray":  # noqa: F821
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        raise ValueError(f"values must be a finite 1-D array, got {v.shape}")
    return np.where(v > float(above), float(weight), 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# LLM content labels on a seeded sample (measurement)
# ---------------------------------------------------------------------------------------------
SAFETY_CATEGORIES = ("sexual", "profanity", "violence", "hate", "drugs")

SAFETY_RUBRICS = {
    "safety_v1": {
        "instruction": """You review training data for a speech AI model. Below is an excerpt of a real, unscripted podcast conversation between speaker A and speaker B (an automatic transcript: some words may be misrecognised). Rate what EACH speaker says, separately, on five categories, 0 / 1 / 2:

- sexual: 0 none; 1 mentions sex or sexuality in passing, jokes about it non-graphically; 2 explicit or graphic sexual content (describing sex acts, genitals, porn in detail).
- profanity: 0 none; 1 occasional swearing; 2 heavy swearing or swearing aimed at someone as an insult.
- violence: 0 none; 1 talks about violence, fighting, weapons or war; 2 graphic description of injury, gore or killing, or endorsing violence against someone.
- hate: 0 none; 1 a demeaning generalisation about a group defined by race, religion, gender, sexuality, disability or nationality; 2 a slur or a hateful attack on such a group.
- drugs: 0 none; 1 talks about drugs or drug use; 2 describes using illegal drugs approvingly, or how to obtain or use them.

Rate only what that speaker says, not the topic of the other speaker. A speaker with no words gets all zeros.

Answer with JSON only: {{"A": {{"sexual": 0, "profanity": 0, "violence": 0, "hate": 0, "drugs": 0}}, "B": {{"sexual": 0, "profanity": 0, "violence": 0, "hate": 0, "drugs": 0}}}}

TRANSCRIPT:
{transcript}""",
    },
}

_SIDE_SCHEMA = {
    "type": "object",
    "properties": {c: {"type": "integer", "enum": [0, 1, 2]} for c in SAFETY_CATEGORIES},
    "required": list(SAFETY_CATEGORIES),
    "additionalProperties": False,
}
_SAFETY_JSON_SCHEMA = {
    "type": "object",
    "properties": {"A": _SIDE_SCHEMA, "B": _SIDE_SCHEMA},
    "required": ["A", "B"],
    "additionalProperties": False,
}


def safety_rubric_digest(name: str) -> str:
    return hashlib.sha256(json.dumps(SAFETY_RUBRICS[name], sort_keys=True).encode()).hexdigest()[:16]


def _ab_transcript(wa: list, wb: list, gap: float = 1.5) -> str:
    """Both speakers as ``A:`` / ``B:`` utterances (a speaker's words < ``gap`` s apart), ordered by onset."""
    utts = []
    for who, ws in (("A", wa), ("B", wb)):
        cur = None
        for w in sorted(ws or [], key=lambda w: float(w["start"])):
            if cur and float(w["start"]) - cur["end"] < gap:
                cur["words"].append(str(w["text"]))
                cur["end"] = float(w["end"])
            else:
                cur = {"who": who, "start": float(w["start"]), "end": float(w["end"]), "words": [str(w["text"])]}
                utts.append(cur)
    utts.sort(key=lambda u: u["start"])
    return "\n".join(f"{u['who']}: {' '.join(u['words'])}" for u in utts)


def sample_rows(num_rows: int, n: int, seed: int) -> list:
    """The seeded sample: ``numpy.random.default_rng(seed).choice(num_rows, n, replace=False)``, sorted."""
    import numpy as np

    return sorted(int(i) for i in np.random.default_rng(int(seed)).choice(int(num_rows), int(n), replace=False))


def safety_label_items(data_path: str, rows: list) -> list:
    """The items to label: per sampled table row its id, both sides' text (A = the stored assistant, B = the other)
    and the A/B transcript."""
    from datasets import load_from_disk

    ds = load_from_disk(str(data_path))
    cols = ["id", "alignments"] + [c for c in ("alignments_other", "other_ok") if c in ds.column_names]
    sub = ds.select(rows).select_columns(cols)
    items = []
    for i, r in zip(rows, sub):
        wa, wb = r["alignments"] or [], r.get("alignments_other") or []
        items.append(
            {
                "row": int(i),
                "id": r["id"],
                "other_ok": bool(r.get("other_ok", False)),
                "text_a": _side_text(wa),
                "text_b": _side_text(wb),
                "transcript": _ab_transcript(wa, wb),
            }
        )
    return items


def label_safety(items: list, url: str, llm_name: str, rubric: str, concurrency: int = 64, progress=None) -> list:
    """One LLM call per item (seeded by the transcript, temperature 0, guided JSON, 5 tries): ``{"A": {...},
    "B": {...}}`` or ``{"error": ...}``."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url)
    instr = SAFETY_RUBRICS[rubric]["instruction"]

    def one(it):
        msg = instr.format(transcript=it["transcript"] or "(no words)")
        seed = int(hashlib.md5(it["transcript"].encode()).hexdigest(), 16) % (2**31)
        err = ""
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=llm_name,
                    messages=[{"role": "user", "content": msg}],
                    seed=(seed + attempt) % (2**31),
                    temperature=0.0,
                    max_tokens=4096,  # GPT-OSS reasons before it answers
                    extra_body={"guided_json": _SAFETY_JSON_SCHEMA},
                )
                raw = (resp.choices[0].message.content or "").strip()
                parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
                out = {s: {c: int(parsed[s][c]) for c in SAFETY_CATEGORIES} for s in ("A", "B")}
                if any(v not in (0, 1, 2) for s in out.values() for v in s.values()):
                    raise ValueError(f"label out of range: {out}")
                return out
            except Exception as e:  # API error, empty or unparsable reply: retry
                err = f"{type(e).__name__}: {e}"[:200]
        return {"error": err}

    return map_concurrent(one, items, concurrency=concurrency, progress_path=progress)


def safety_sample_summary(items: list, labels: list, lexicon: str = "explicit_v1") -> dict:
    """Prevalence per category and side (share of windows with label >= 1 and == 2), and the lexicon's agreement with
    the LLM on the same windows (per side: precision / recall of "lexicon hit" vs "LLM label >= 1")."""
    pats = lexicon_patterns(lexicon)
    ok = [(it, lab) for it, lab in zip(items, labels) if "error" not in lab]
    out = {"n": len(items), "failed": len(items) - len(ok), "sides": {}}
    lex_map = {"sexual": "sexual", "profanity": "profanity", "hate": "slur"}
    for side, key, tkey in (("A (stored assistant)", "A", "text_a"), ("B (other)", "B", "text_b")):
        pool = [(it, lab) for it, lab in ok if (key == "A" or it["other_ok"] or it[tkey])]
        d = {"n": len(pool)}
        for c in SAFETY_CATEGORIES:
            ge1 = sum(lab[key][c] >= 1 for _, lab in pool)
            eq2 = sum(lab[key][c] == 2 for _, lab in pool)
            d[c] = {"ge1": ge1, "ge1_share": round(ge1 / max(len(pool), 1), 4), "eq2": eq2}
            if c in lex_map:
                lx = [lexicon_hits(it[tkey], pats)[lex_map[c]] > 0 for it, _ in pool]
                llm = [lab[key][c] >= 1 for _, lab in pool]
                tp = sum(a and b for a, b in zip(lx, llm))
                d[c]["lexicon"] = {
                    "hits": sum(lx),
                    "precision_vs_llm_ge1": round(tp / sum(lx), 3) if sum(lx) else None,
                    "recall_vs_llm_ge1": round(tp / sum(llm), 3) if sum(llm) else None,
                    "recall_vs_llm_eq2": (
                        round(
                            sum(a and lab[key][c] == 2 for a, (_, lab) in zip(lx, pool))
                            / max(1, sum(lab[key][c] == 2 for _, lab in pool)),
                            3,
                        )
                    ),
                }
        out["sides"][side] = d
    # a window where a TRAINABLE side (A always, B when other_ok) is labelled 2 on any category
    out["trainable_eq2_any"] = sum(
        any(lab["A"][c] == 2 for c in SAFETY_CATEGORIES)
        or (it["other_ok"] and any(lab["B"][c] == 2 for c in SAFETY_CATEGORIES))
        for it, lab in ok
    )
    out["trainable_sexual_ge1"] = sum(
        lab["A"]["sexual"] >= 1 or (it["other_ok"] and lab["B"]["sexual"] >= 1) for it, lab in ok
    )
    out["trainable_sexual_eq2"] = sum(
        lab["A"]["sexual"] == 2 or (it["other_ok"] and lab["B"]["sexual"] == 2) for it, lab in ok
    )
    return out


class SafetyLabelSample(Job):
    """LLM content labels (sexual / profanity / violence / hate / drugs, 0-2, per speaker) on a seeded sample of a
    window corpus: ``n`` table rows of ``data`` (:func:`sample_rows`, ``seed``), each labelled from its two-sided
    transcript (A = the stored assistant, B = the other side) by ``llm_name`` under ``rubric``
    (:data:`SAFETY_RUBRICS`, digest hashed). A MEASUREMENT (how common is it, and how well does the lexicon find it),
    not a filter: labelling a whole corpus this way costs ~135x this job.

    Outputs: ``out_rows`` (jsonl: row, id, other_ok, both sides' text, labels) and ``out_summary``
    (:func:`safety_sample_summary`). More than 2% failed calls is fatal."""

    def __init__(
        self,
        *,
        data: tk.Path,
        n: int = 1000,
        seed: int = 0,
        llm_name: str = "openai/gpt-oss-120b",
        rubric: str = "safety_v1",
    ):
        assert rubric in SAFETY_RUBRICS, rubric
        self.data = data
        self.n = int(n)
        self.seed = int(seed)
        self.llm_name = llm_name
        self.rubric = rubric
        self.rubric_digest = safety_rubric_digest(rubric)
        self.out_rows = self.output_path("rows.jsonl")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        from sisyphus import global_settings as gs

        try:
            d = json.load(open(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress.json")))
            return min(d["done"] / d["total"], 1.0)
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return None

    def run(self):
        run_safety_sample(
            self.data.get_path(),
            self.n,
            self.seed,
            self.llm_name,
            self.rubric,
            self.out_rows.get_path(),
            self.out_summary.get_path(),
            progress=os.path.abspath("progress.json"),
        )


def run_safety_sample(data_path, n, seed, llm_name, rubric, out_rows, out_summary, progress=None) -> dict:
    """:class:`SafetyLabelSample`'s worker (also run standalone by ``analysis/row_filter_stats.py``'s sbatch)."""
    from datasets import load_from_disk

    num = load_from_disk(str(data_path)).num_rows
    rows = sample_rows(num, n, seed)
    items = safety_label_items(data_path, rows)
    print(f"[safety] {len(items)} windows of {num} (seed {seed}), {llm_name}, rubric {rubric}", flush=True)
    with vllm_server(llm_name) as url:
        labels = label_safety(items, url, llm_name, rubric, progress=progress)
    failed = sum("error" in lab for lab in labels)
    print(f"[safety] {failed}/{len(items)} calls failed", flush=True)
    if failed > 0.02 * len(items):
        raise RuntimeError(f"safety labelling failed on {failed}/{len(items)} calls (> 2%)")
    with open(out_rows, "w") as f:
        for it, lab in zip(items, labels):
            f.write(json.dumps({**it, "labels": lab}, ensure_ascii=False) + "\n")
    summary = {
        "population": str(data_path),
        "num_rows": num,
        "seed": seed,
        "llm_name": llm_name,
        "rubric": rubric,
        "rubric_digest": safety_rubric_digest(rubric),
        **safety_sample_summary(items, labels),
    }
    json.dump(summary, open(out_summary, "w"), indent=2)
    print(json.dumps(summary, indent=1), flush=True)
    return summary
