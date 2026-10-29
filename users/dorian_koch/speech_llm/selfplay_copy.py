"""How much of a self-play conversation is copied VERBATIM from the training transcripts (2026-09-28:
much of the training data is one podcast, so verbatim replication is a real risk).

``SelfPlayCopyStats``: for every generated channel of every self-play run (the model's own TEXT stream and the ASR
of its audio, both), each word n-gram (default 8, lower-cased, ``[a-z0-9']`` tokens) is looked up in three nested
sources, all built from the TRAINING table's own word alignments (``alignments`` + ``alignments_other``, both
sides, which is what the text stream is trained on):

  W  the self-play input WINDOW's own real transcript (both sides; for a custom-prompt row, the window its voice
     clip came from),
  E  every training row of the same EPISODE (so ``E - W`` = other windows of that episode),
  O  every training row of OTHER episodes (the whole corpus index; a hash -> episodes table).

Per run: the share of generated n-grams in W / E-only / O-only / none (bootstrap 95% CI over conversations), the
longest verbatim run per channel (consecutive n-grams all found; length = run + n - 1 words) against W, E and O, the
share of channels with a run >= 12 / >= 20 words, and the longest examples verbatim with their source row (a run
is re-checked as ONE contiguous span in that row; ``contiguous=False`` marks a run stitched from pieces). The same
statistics for the REAL input windows against O (other episodes) are the base rate of shared phrases in natural
JRE speech. Controls are other runs: the released model (never trained on JRE) and a model whose training excluded
the windows' episodes (p6).
"""

from __future__ import annotations

import json
import re

from sisyphus import Job, Task, tk
from .label_free_hash import hash_bytes, label_free_hash, values_canon  # noqa: F401

_TOK = re.compile(r"[a-z0-9']+")
_P = 1000003  # rolling-hash multiplier (uint64 arithmetic wraps)


def tokens(text: str) -> list[str]:
    return _TOK.findall(str(text).lower().replace("’", "'"))


def ngram_hashes(ids, n: int):
    """uint64 hash of every n-gram of the int sequence ``ids`` (polynomial, wrapping); empty if shorter."""
    import numpy as np

    a = np.asarray(ids, dtype=np.uint64)
    if len(a) < n:
        return np.zeros(0, np.uint64)
    h = np.zeros(len(a) - n + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * np.uint64(_P) + a[j : j + len(h)] + np.uint64(1)
    return h


def longest_run(mask) -> tuple[int, int]:
    """(start, length) of the longest run of True in ``mask`` (length 0 if none)."""
    best, start, cur, cur_s = 0, 0, 0, 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, start = cur, cur_s
        else:
            cur = 0
    return start, best


def contains(hay, needle) -> bool:
    """``needle`` (a token-id sequence) occurs contiguously in ``hay``."""
    if len(needle) == 0:
        return True
    s = " " + " ".join(map(str, hay)) + " "
    return (" " + " ".join(map(str, needle)) + " ") in s


class SelfPlayCopyStats(Job):
    """Verbatim-copy statistics of self-play runs against a training table (module docstring).

    ``runs``: ``{label: SelfPlayTranscribe dataset}``; ``windows``: the self-play INPUT dataset holding the JRE
    windows' ``id`` / ``alignments`` / ``alignments_other`` (row ids of the runs); ``custom_inputs``: the
    ``CustomSelfPlayPrompts`` dataset (``id`` -> ``voice_window``) for custom-prompt runs; ``corpus``: the training
    table (``id`` = ``<episode>#...``, ``alignments``, ``alignments_other``). Outputs ``summary.json`` and
    ``examples.jsonl`` (the ``n_examples`` longest runs per label and source)."""

    def __init__(
        self,
        *,
        runs: dict,
        windows: tk.Path,
        corpus: tk.Path,
        custom_inputs: tk.Path | None = None,
        n: int = 8,
        n_examples: int = 5,
    ):
        self.runs = {str(k): v for k, v in sorted(runs.items())}
        self.windows = windows
        self.corpus = corpus
        self.custom_inputs = custom_inputs
        self.n = int(n)
        self.n_examples = int(n_examples)
        self.out_summary = self.output_path("summary.json")
        self.out_examples = self.output_path("examples.jsonl")

    @classmethod
    def hash(cls, parsed_args):
        """Run labels are not hashed for a new job (``label_free_hash``)."""
        return label_free_hash(
            cls,
            parsed_args,
            canon=lambda a: {**a, "runs": values_canon(a["runs"])},
            base_hash=lambda a: super(SelfPlayCopyStats, cls).hash(a),
            labels=lambda a: sorted(a["runs"]),
        )

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 48, "time": 4})

    def run(self):
        import numpy as np
        import pyarrow.compute as pc
        from datasets import load_from_disk

        n = self.n
        vocab: dict[str, int] = {}

        def ids_of(toks):
            return np.asarray([vocab.setdefault(t, len(vocab)) for t in toks], dtype=np.int64)

        def ep(rid):
            return str(rid).split("#", 1)[0]

        # ---- the corpus index: every n-gram of every side of every training row ----
        tbl = load_from_disk(self.corpus.get_path()).select_columns(["id", "alignments", "alignments_other"])
        row_ids = list(tbl["id"])
        episodes = sorted({ep(r) for r in row_ids})
        ep_idx = {e: i for i, e in enumerate(episodes)}
        row_tokens = {}  # row -> (token ids side a, side b), int32
        hs, es, rs = [], [], []
        at = tbl.data.table if hasattr(tbl.data, "table") else tbl.data
        texts_by_col = {}
        for col in ("alignments", "alignments_other"):
            ca = at.column(col).combine_chunks()
            flat = pc.list_flatten(ca).field("text").to_pylist()
            off = ca.offsets.to_numpy()
            texts_by_col[col] = (flat, off)
        for r, rid in enumerate(row_ids):
            e = ep_idx[ep(rid)]
            sides = []
            for col in ("alignments", "alignments_other"):
                flat, off = texts_by_col[col]
                ids = ids_of(tokens(" ".join(t or "" for t in flat[off[r] : off[r + 1]])))
                sides.append(ids.astype(np.int32))
                h = ngram_hashes(ids, n)
                hs.append(h)
                es.append(np.full(len(h), e, np.int32))
                rs.append(np.full(len(h), r, np.int32))
            row_tokens[r] = sides
            if r % 20000 == 0:
                print(f"[copy] corpus row {r}/{len(row_ids)}, vocab {len(vocab)}", flush=True)
        del texts_by_col
        h_all, e_all, r_all = np.concatenate(hs), np.concatenate(es), np.concatenate(rs)
        del hs, es, rs
        order = np.lexsort((e_all, h_all))
        h_s, e_s, r_s = h_all[order], e_all[order], r_all[order]
        del h_all, e_all, r_all, order
        uh, first, counts = np.unique(h_s, return_index=True, return_counts=True)
        last = first + counts - 1
        first_ep, last_ep = e_s[first], e_s[last]
        first_row, last_row = r_s[first], r_s[last]
        print(
            f"[copy] corpus: {len(row_ids)} rows, {len(episodes)} episodes, {len(h_s)} {n}-grams, {len(uh)} distinct",
            flush=True,
        )
        # per-episode n-gram sets for the episodes the runs touch (filled lazily)
        ep_rows: dict[int, list[int]] = {}
        for r, rid in enumerate(row_ids):
            ep_rows.setdefault(ep_idx[ep(rid)], []).append(r)
        ep_sets: dict[int, set] = {}

        def ep_set(e):
            if e not in ep_sets:
                s = set()
                for r in ep_rows.get(e, []):
                    for ids in row_tokens[r]:
                        s.update(ngram_hashes(ids, n).tolist())
                ep_sets[e] = s
            return ep_sets[e]

        def lookup(h, e):
            """-> (in corpus at all, in an episode other than e) per hash."""
            i = np.searchsorted(uh, h)
            i = np.minimum(i, len(uh) - 1)
            found = (len(uh) > 0) & (uh[i] == h)
            other = found & ((first_ep[i] != e) | (last_ep[i] != e))
            return found, other, i

        # ---- the windows (real transcripts) ----
        wins = load_from_disk(self.windows.get_path()).select_columns(["id", "alignments", "alignments_other"])
        win = {}
        for r in wins:
            win[r["id"]] = [
                ids_of(tokens(" ".join(w["text"] for w in (r["alignments"] or [])))),
                ids_of(tokens(" ".join(w["text"] for w in (r["alignments_other"] or [])))),
            ]
        voice_window = {}
        if self.custom_inputs is not None:
            for r in load_from_disk(self.custom_inputs.get_path()).select_columns(["id", "voice_window"]):
                voice_window[r["id"]] = r["voice_window"]

        def window_of(rid):
            w = voice_window.get(rid, rid)
            if w not in win:
                raise RuntimeError(f"row {rid}: window {w} not in the windows dataset")
            return w

        def channel_stats(toks, wid, examples, meta):
            e = ep_idx.get(ep(wid), -1)
            tid = ids_of(toks)
            h = ngram_hashes(tid, n)
            if len(h) == 0:
                return {"ngrams": 0, "W": 0, "E_only": 0, "O_only": 0, "run_W": 0, "run_E": 0, "run_O": 0, "run_any": 0}
            wset = set()
            for side in win[wid]:
                wset.update(ngram_hashes(side, n).tolist())
            es_ = ep_set(e) if e >= 0 else set()
            hl = h.tolist()
            in_w = np.array([x in wset for x in hl])
            in_e = np.array([x in es_ for x in hl]) | in_w
            found, in_o, idx = lookup(h, e)
            res = {
                "ngrams": len(h),
                "W": int(in_w.sum()),
                "E_only": int((in_e & ~in_w).sum()),
                "O_only": int((in_o & ~in_e).sum()),
            }
            for key, mask in (("W", in_w), ("E", in_e), ("O", in_o), ("any", in_e | in_o)):
                s, L = longest_run(mask.tolist())
                span = L + n - 1 if L else 0
                res[f"run_{key}"] = span
                if key in ("W", "E", "O") and L:
                    words = toks[s : s + span]
                    if key == "O":
                        k = idx[s]
                        src_row = int(first_row[k] if first_ep[k] != e else last_row[k])
                        hay = row_tokens[src_row]
                        src = row_ids[src_row]
                    elif key == "E":
                        hay = [t for r in ep_rows.get(e, []) for t in row_tokens[r]]
                        src = f"episode {ep(wid)}"
                    else:
                        hay = win[wid]
                        src = f"window {wid}"
                    contig = any(contains(side, tid[s : s + span]) for side in hay)
                    examples.append(
                        {
                            **meta,
                            "against": key,
                            "words": span,
                            "text": " ".join(words),
                            "source": src,
                            "contiguous": contig,
                        }
                    )
            return res

        summary = {"n": n, "corpus_rows": len(row_ids), "corpus_episodes": len(episodes), "labels": {}}
        all_examples = []

        def aggregate(per_conv):
            """per_conv: list of per-conversation dicts summed over its channels."""
            rng = np.random.default_rng(0)
            tot = {k: np.array([c[k] for c in per_conv], float) for k in ("ngrams", "W", "E_only", "O_only")}
            out = {"n_conv": len(per_conv), "ngrams": int(tot["ngrams"].sum())}
            for k in ("W", "E_only", "O_only"):
                share = tot[k].sum() / max(tot["ngrams"].sum(), 1)
                bs = []
                for _ in range(2000):
                    i = rng.integers(0, len(per_conv), len(per_conv))
                    bs.append(tot[k][i].sum() / max(tot["ngrams"][i].sum(), 1))
                out[f"share_{k}"] = float(share)
                out[f"share_{k}_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            return out

        def run_stats(channels):
            out = {}
            for key in ("W", "E", "O", "any"):
                runs = np.array([c[f"run_{key}"] for c in channels], float)
                out[f"run_{key}"] = {
                    "max": int(runs.max()) if len(runs) else 0,
                    "median": float(np.median(runs)) if len(runs) else 0.0,
                    "frac_ge12": float(np.mean(runs >= 12)) if len(runs) else 0.0,
                    "frac_ge20": float(np.mean(runs >= 20)) if len(runs) else 0.0,
                }
            return out

        for label, path in self.runs.items():
            ds = load_from_disk(path.get_path()).select_columns(["id", "text_a", "text_b", "words_a", "words_b"])
            entry = {}
            for source in ("text", "asr"):
                per_conv, channels, examples = [], [], []
                for r in ds:
                    wid = window_of(r["id"])
                    conv = {"ngrams": 0, "W": 0, "E_only": 0, "O_only": 0}
                    for side in ("a", "b"):
                        if source == "text":
                            toks = tokens(r[f"text_{side}"])
                        else:
                            toks = tokens(" ".join(w["text"] for w in json.loads(r[f"words_{side}"])))
                        meta = {"label": label, "source_kind": source, "id": r["id"], "channel": side}
                        c = channel_stats(toks, wid, examples, meta)
                        channels.append(c)
                        for k in conv:
                            conv[k] += c[k]
                    per_conv.append(conv)
                entry[source] = {**aggregate(per_conv), **run_stats(channels)}
                for key in ("W", "E", "O"):
                    ex = sorted((x for x in examples if x["against"] == key), key=lambda x: -x["words"])
                    all_examples += ex[: self.n_examples]
            summary["labels"][label] = entry
            t = entry["text"]
            print(
                f"[copy] {label}: text share W {t['share_W']:.4f} E-only {t['share_E_only']:.4f} O-only "
                f"{t['share_O_only']:.4f}; longest W/E/O {t['run_W']['max']}/{t['run_E']['max']}/{t['run_O']['max']} words",
                flush=True,
            )

        # ---- the real windows the runs used, against OTHER episodes: the natural base rate ----
        used = sorted(
            {window_of(r) for label, path in self.runs.items() for r in load_from_disk(path.get_path())["id"]}
        )
        per_conv, channels, examples = [], [], []
        for wid in used:
            conv = {"ngrams": 0, "W": 0, "E_only": 0, "O_only": 0}
            e = ep_idx.get(ep(wid), -1)
            for side in win[wid]:
                h = ngram_hashes(side, n)
                found, in_o, idx = lookup(h, e)
                conv["ngrams"] += len(h)
                conv["O_only"] += int(in_o.sum())
                s, L = longest_run(in_o.tolist())
                channels.append({"run_W": 0, "run_E": 0, "run_O": L + n - 1 if L else 0, "run_any": 0})
            per_conv.append(conv)
        real = {**aggregate(per_conv), **run_stats(channels)}
        real["windows"] = len(used)
        # coverage: how much of each real window's text the training table has in that episode (is the window's audio
        # in the corpus at all?)
        cov = []
        for wid in used:
            e = ep_idx.get(ep(wid), -1)
            es_ = ep_set(e) if e >= 0 else set()
            h = [x for side in win[wid] for x in ngram_hashes(side, n).tolist()]
            if h:
                cov.append(float(np.mean([x in es_ for x in h])))
        real["episode_coverage_mean"] = float(np.mean(cov)) if cov else None
        real["episode_coverage_frac_ge50"] = float(np.mean(np.array(cov) >= 0.5)) if cov else None
        summary["real_windows_vs_other_episodes"] = real
        print(
            f"[copy] real windows ({len(used)}) vs other episodes: share {real['share_O_only']:.4f}, longest "
            f"{real['run_O']['max']}; corpus coverage of the windows' own text {real['episode_coverage_mean']}",
            flush=True,
        )
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        with open(self.out_examples.get_path(), "w") as f:
            for x in all_examples:
                f.write(json.dumps(x) + "\n")
