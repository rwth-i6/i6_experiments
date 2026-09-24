"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/d18_beyond_per_jobs.py (js_divergence :80,
js_rows :98, speaker_counts :113, bootstrap_weights :136, row_contrast :292 and the B3 part of
decode_stats :240, as the lean reader :class:`JsRowsReadJob`; B1 and B4 of ``BeyondPerReadJob`` are
not ported).

The D18 B3 JS rows: :data:`JS_ROWS_CONVENTION` (below) is that reader's registered rule.

Not ported: i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/ngram_mode_seeking.py (the
n-gram mode-seeking check ``NgramModeSeekingJob`` and its helpers).  No phase-4a run uses it; its
rows need a GAN decode.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from sisyphus import Job, Task, tk

from ..phones import ARPABET_39

__all__ = ["js_divergence", "js_rows", "speaker_counts", "bootstrap_weights", "decode_b3", "row_contrast",
           "JsRowsReadJob", "JS_ROWS_CONVENTION"]


JS_ROWS_CONVENTION = (
    "D18 beyond-PER convention, B3 only (SAE_4A_lexlat.md D18, registered before any read; descriptive, NO GATE):\n"
    "  * Plain PER keeps its registered role.  These reads EXPLAIN a PER difference or a PER tie;\n"
    "    they never decide, never gate, never select.  Gold phones are an analysis reference only.\n"
    "  * B3, output distribution, from each dev-other greedy decode.  JS = Jensen-Shannon divergence,\n"
    "    log base 2 (0..1), no smoothing: JS(P, Q) = KL(P||M)/2 + KL(Q||M)/2, M = (P + Q)/2.\n"
    "    Unigram = phone type frequencies; bigram = joint frequencies of adjacent pairs within an\n"
    "    utterance (no boundary symbols).  Against GOLD and between runs: SIL-free strings\n"
    "    (greedy_phones), 39 types.  Against the TEXT PRIOR: strings with SIL (greedy_raw), 40 types,\n"
    "    against the prior's own targets (agg.text_target_counts: unigram, joint p(h) p(k|h)).\n"
    "    Types used = non-SIL types emitted at least once; SIL share = SIL tokens / raw tokens.\n"
    "    Per registered PER row (cand vs base at the row's epoch): dJS = JS(cand, ref) - JS(base, ref)\n"
    "    for ref = gold and prior, unigram and bigram, with a speaker-clustered 95% interval (2000\n"
    "    resamples, seed 0, speaker = tag prefix before the first '-'; the speaker draw of\n"
    "    d8_admission.cluster_bootstrap; cand, base and gold counts from the same resampled speakers,\n"
    "    the prior fixed); JS(cand, base) as a point value.\n"
)


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence in bits between two count / probability vectors (normalized here)."""
    import numpy as np

    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    assert p.shape == q.shape and (p >= 0).all() and (q >= 0).all(), (p.shape, q.shape)
    assert p.sum() > 0 and q.sum() > 0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / m[nz])).sum())

    return 0.5 * _kl(p) + 0.5 * _kl(q)


def js_rows(p, q):
    """Row-wise :func:`js_divergence` of two [B, K] count matrices (q may be one [K] vector)."""
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    q = np.broadcast_to(np.asarray(q, dtype=np.float64), p.shape)
    p = p / p.sum(1, keepdims=True)
    q = q / q.sum(1, keepdims=True)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log2(p / m), 0.0).sum(1)
        kq = np.where(q > 0, q * np.log2(q / m), 0.0).sum(1)
    return 0.5 * kp + 0.5 * kq


def speaker_counts(strings: Dict[str, Sequence[str]], clusters, tags: Sequence[str],
                   types: Sequence[str]):
    """Per-speaker unigram [S, K] and flattened bigram [S, K*K] counts over ``types``.

    ``clusters`` are position arrays into ``tags`` (``s1a_job._clusters_by_speaker``).  A symbol
    outside ``types`` raises: a SIL-free count must never silently drop a SIL.
    """
    import numpy as np

    index = {t: i for i, t in enumerate(types)}
    k = len(types)
    uni = np.zeros((len(clusters), k), dtype=np.float64)
    bi = np.zeros((len(clusters), k * k), dtype=np.float64)
    for s, positions in enumerate(clusters):
        for pos in positions:
            ids = [index[p] for p in strings[tags[int(pos)]]]
            for i in ids:
                uni[s, i] += 1
            for a, b in zip(ids[:-1], ids[1:]):
                bi[s, a * k + b] += 1
    return uni, bi


def bootstrap_weights(n_clusters: int, *, n_boot: int, seed: int):
    """[n_boot, n_clusters] speaker multiplicities, drawn exactly as ``cluster_bootstrap`` draws them.

    ``d8_admission.cluster_bootstrap`` draws ``rng.integers(0, n, size=n)`` per resample from
    ``np.random.default_rng(seed)`` and takes every row of each picked speaker; a count statistic
    over those rows is ``weights @ per-speaker counts``.
    """
    import numpy as np

    rng = np.random.default_rng(int(seed))
    w = np.zeros((int(n_boot), int(n_clusters)), dtype=np.float64)
    for b in range(int(n_boot)):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        w[b] = np.bincount(picked, minlength=n_clusters)
    return w


def decode_b3(*, raw: Dict[str, List[str]], phones: Dict[str, List[str]], tags: Sequence[str], clusters,
              gold_uni, gold_bi, prior_uni, prior_bi) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The B3 point statistics of one decode plus its per-speaker counts for the row intervals.

    The B3 half of the source's ``decode_stats``.  The source counted SIL and raw tokens on the
    argmax runs of the posteriors after asserting those runs equal ``greedy_raw``; this reader has
    no posteriors and counts the same tokens on ``greedy_raw`` itself.
    """
    from ..phones import PHONES, SIL

    raw_tokens = sil_tokens = 0
    for tag in tags:
        assert [p for p in raw[tag] if p != SIL] == list(phones[tag]), tag
        sil_tokens += sum(1 for p in raw[tag] if p == SIL)
        raw_tokens += len(raw[tag])
    uni39, bi39 = speaker_counts(phones, clusters, tags, ARPABET_39)
    uni40, bi40 = speaker_counts(raw, clusters, tags, PHONES)
    u39, b39, u40, b40 = uni39.sum(0), bi39.sum(0), uni40.sum(0), bi40.sum(0)
    b3 = {
        "js_gold_unigram": js_divergence(u39, gold_uni), "js_gold_bigram": js_divergence(b39, gold_bi),
        "js_prior_unigram": js_divergence(u40, prior_uni), "js_prior_bigram": js_divergence(b40, prior_bi),
        "types_used": int((u39 > 0).sum()), "sil_share": sil_tokens / raw_tokens,
        "raw_tokens": raw_tokens, "sil_tokens": sil_tokens, "phone_tokens": int(u39.sum()),
    }
    counts = {"uni39": uni39, "bi39": bi39, "uni40": uni40, "bi40": bi40}
    return b3, counts


def row_contrast(cand: Dict[str, Any], base: Dict[str, Any], gold: Dict[str, Any], prior_uni, prior_bi,
                 weights) -> Dict[str, Any]:
    """dJS = JS(cand, ref) - JS(base, ref) with speaker-clustered intervals; JS(cand, base) point."""
    import numpy as np

    ones = np.ones((1, weights.shape[1]))
    out: Dict[str, Any] = {}
    for ref_name, key_c, ref_uni, ref_bi in (("gold", "39", gold["uni39"], gold["bi39"]),
                                            ("prior", "40", None, None)):
        for order in ("unigram", "bigram"):
            short = "uni" if order == "unigram" else "bi"
            c_spk, b_spk = cand[f"{short}{key_c}"], base[f"{short}{key_c}"]

            def delta(w):
                if ref_name == "gold":
                    ref = w @ (ref_uni if order == "unigram" else ref_bi)
                else:
                    ref = prior_uni if order == "unigram" else prior_bi
                return js_rows(w @ c_spk, ref) - js_rows(w @ b_spk, ref)

            point = float(delta(ones)[0])
            boot = delta(weights)
            lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            out[f"djs_{ref_name}_{order}"] = {"mean": point, "ci95": [lo, hi],
                                              "excludes_zero": bool(lo > 0 or hi < 0)}
    out["js_cand_base_unigram"] = js_divergence(cand["uni39"].sum(0), base["uni39"].sum(0))
    out["js_cand_base_bigram"] = js_divergence(cand["bi39"].sum(0), base["bi39"].sum(0))
    return out


class JsRowsReadJob(Job):
    """D18 B3 for a set of greedy decodes: per-decode JS statistics and the registered row contrasts.

    The lean part of the source's ``BeyondPerReadJob``: B3 only (no B1 convergence, no B4 duration,
    no posteriors).  Every B3 number is computed by the source's own functions on the same inputs.
    The reporting rule is :data:`JS_ROWS_CONVENTION`, printed at the head of the report.

    :param decodes: ``{name: {epoch: {"raw": greedy_raw.json, "phones": greedy_phones.json,
        "per": per.json (optional, only echoed)}}}``, the ``BlankfreeGreedyPerJob`` outputs.
    :param rows: ``({"tag", "cand", "cand_epoch", "base", "base_epoch"}, ...)``, each contrasted as
        ``cand - base``.
    :param gold: GoldPhonesJob json; :param split: its split; :param expected_utterances: asserted.
    :param prior_npz: the text prior (``PhoneNgramPriorJob``'s ``prior.npz``).
    :param n_boot: / :param seed: the speaker-bootstrap resamples and seed (the convention registers
        2000 at seed 0; both are required arguments, as in the source).
    """

    def __init__(self, *, decodes: Dict[str, Dict[int, Dict[str, Any]]], rows: Sequence[Dict[str, Any]],
                 gold: tk.Path, split: str, expected_utterances: int, prior_npz: tk.Path,
                 n_boot: int, seed: int):
        super().__init__()
        self.decodes = decodes
        self.rows = tuple(dict(r) for r in rows)
        self.gold, self.split = gold, str(split)
        self.expected_utterances = int(expected_utterances)
        self.prior_npz = prior_npz
        self.n_boot, self.seed = int(n_boot), int(seed)
        for name, run in decodes.items():
            assert run, name
        for row in self.rows:
            assert row["cand_epoch"] in decodes[row["cand"]], row
            assert row["base_epoch"] in decodes[row["base"]], row
        self.out_json = self.output_path("js_rows.json")
        self.out_report = self.output_path("report.txt")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from ..model import agg as agg_mod
        from ..model.prior import PhoneNgramPrior
        from .gaps import _clusters_by_speaker

        def _p(path):
            return tk.uncached_path(path)

        def _json(path):
            with open(_p(path)) as fh:
                return json.load(fh)

        lines = [JS_ROWS_CONVENTION, ""]
        out: Dict[str, Any] = {"convention": JS_ROWS_CONVENTION, "n_boot": self.n_boot, "seed": self.seed}

        gold = _json(self.gold)[self.split]
        tags = sorted(gold)
        assert len(tags) == self.expected_utterances, (len(tags), self.expected_utterances)
        clusters = _clusters_by_speaker(tags)
        prior = PhoneNgramPrior.load(_p(self.prior_npz))
        pu, pb = agg_mod.text_target_counts(prior)
        prior_uni = pu.double().numpy()
        prior_bi = pb.double().numpy().reshape(-1)
        g_uni, g_bi = speaker_counts(gold, clusters, tags, ARPABET_39)
        gold_counts = {"uni39": g_uni, "bi39": g_bi}
        weights = bootstrap_weights(len(clusters), n_boot=self.n_boot, seed=self.seed)
        out["references"] = {"n_speakers": len(clusters), "n_utterances": len(tags)}
        lines.append(f"{len(tags)} utterances, {len(clusters)} speakers")

        decodes: Dict[str, Dict[int, Any]] = {}
        counts: Dict[Tuple[str, int], Any] = {}
        lines.append("\nB3 PER DECODE (greedy; JS in bits)")
        lines.append("  run                  ep     PER  JSg_uni  JSg_bi JSp_uni  JSp_bi types  SIL%")
        for name, run in self.decodes.items():
            decodes[name] = {}
            for epoch in sorted(run):
                d = run[epoch]
                raw, phones = _json(d["raw"]), _json(d["phones"])
                assert set(raw) == set(phones) == set(tags), (name, epoch)
                b3, cnt = decode_b3(raw=raw, phones=phones, tags=tags, clusters=clusters,
                                    gold_uni=g_uni.sum(0), gold_bi=g_bi.sum(0),
                                    prior_uni=prior_uni, prior_bi=prior_bi)
                stats = {"b3": b3}
                if d.get("per") is not None:
                    stats["per"] = float(_json(d["per"])["per"])
                decodes[name][epoch] = stats
                counts[(name, epoch)] = cnt
                per_cell = f"{stats['per']:.4f}" if "per" in stats else "   n/a"
                lines.append(
                    f"  {name:20s} {epoch:2d} {per_cell}  {b3['js_gold_unigram']:.4f}  "
                    f"{b3['js_gold_bigram']:.4f}  {b3['js_prior_unigram']:.4f}  {b3['js_prior_bigram']:.4f} "
                    f"{b3['types_used']:5d} {100 * b3['sil_share']:5.2f}")
        out["decodes"] = {n: {str(e): v for e, v in d.items()} for n, d in decodes.items()}

        out["rows"] = []
        lines.append(f"\nB3 ROWS (cand - base; dJS = JS(cand, ref) - JS(base, ref); 95% speaker-clustered "
                     f"interval, {self.n_boot} resamples, seed {self.seed}; descriptive)")
        for row in self.rows:
            res = row_contrast(counts[(row["cand"], row["cand_epoch"])],
                               counts[(row["base"], row["base_epoch"])], gold_counts, prior_uni,
                               prior_bi, weights)
            out["rows"].append({**row, **res})
            lines.append(f"  {row['tag']}: JS(cand, base) uni {res['js_cand_base_unigram']:.5f} "
                         f"bi {res['js_cand_base_bigram']:.5f}")
            for key in ("djs_gold_unigram", "djs_gold_bigram", "djs_prior_unigram", "djs_prior_bigram"):
                r = res[key]
                lines.append(f"    {key:18s} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]"
                             + ("  excludes 0" if r["excludes_zero"] else ""))

        with open(self.out_json.get_path(), "w") as fh:
            json.dump(out, fh, indent=1)
        with open(self.out_report.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
