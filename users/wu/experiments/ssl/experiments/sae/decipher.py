"""SAE §1a — decipherment primary track, method (i) hard assignment (CPU).

A: V_u -> V_phi recovered *without supervision* from three signals only: unit corpus statistics,
the fixed phoneme n-gram LM (KenLM 4-gram, phoneme_lm.py), and the target text's phone n-gram
distribution. Pipeline per SAE_PLAN §1a(i): CDF-frequency init -> skip-k bigram coordinate descent
-> ICM polish maximizing L(A) = sum_g c(g) log p_4g(A(g)) over unit 4-grams g.

Because the phoneme LM is over *real* phones, there is no label-permutation ambiguity: a correct
decipherment maps each unit to its true phone. So on simulated unit streams (units drawn from 𝒯_φ
phone sequences via a known many-to-one map + fertility + emission noise) we can ground-truth the
recovery exactly: freq-weighted fraction of units u with A(u)==A_true(u), plus decode PER vs the true
deduped phone sequences. This is the reality-anchored CPU check for the whole decipherment track.

Pure/CPU-only (numpy + the ARPA text file); no GPU, no sisyphus dependency for the core so it runs
standalone (`python decipher.py --help`).
"""

from __future__ import annotations

import argparse
import gzip
import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

BOS, EOS = "<s>", "</s>"
NEG_INF = -99.0  # log10 floor


# --------------------------------------------------------------------------------------- ARPA LM

class ArpaLM:
    """Katz backoff n-gram LM read from an ARPA file (base-10 logs, as written by lmplz)."""

    def __init__(self, path: str):
        self.order = 0
        self.ng: List[Dict[Tuple[str, ...], Tuple[float, float]]] = [ {} ]  # index by n; ng[0] unused
        self._memo: Dict[Tuple[Tuple[str, ...], str], float] = {}
        self._load(path)

    def _open(self, path):
        return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

    def _load(self, path: str):
        cur = 0
        with self._open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line[0] == "\\":
                    if line.endswith("-grams:"):
                        cur = int(line[1:].split("-")[0])
                        while len(self.ng) <= cur:
                            self.ng.append({})
                        self.order = max(self.order, cur)
                    elif line == "\\end\\":
                        break
                    continue
                if cur == 0:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                lp = float(parts[0])
                words = tuple(parts[1].split())
                bo = float(parts[2]) if len(parts) >= 3 else 0.0
                if len(words) == cur:
                    self.ng[cur][words] = (lp, bo)

    def cond_logprob(self, hist: Sequence[str], w: str) -> float:
        """log10 p(w | hist) with Katz backoff; hist most-recent-last, capped at order-1. Memoized."""
        hist = tuple(hist)[-(self.order - 1):] if self.order > 1 else ()
        key = (hist, w)
        v = self._memo.get(key)
        if v is None:
            v = self._cond(hist, w)
            self._memo[key] = v
        return v

    def _cond(self, hist: Tuple[str, ...], w: str) -> float:
        ng = hist + (w,)
        d = self.ng[len(ng)]
        hit = d.get(ng)
        if hit is not None:
            return hit[0]
        if not hist:
            uni = self.ng[1]
            hit = uni.get((w,)) or uni.get(("<unk>",))
            return hit[0] if hit is not None else NEG_INF
        bo = self.ng[len(hist)].get(hist)
        bw = bo[1] if bo is not None else 0.0
        return bw + self._cond(hist[1:], w)

    def score_seq(self, tokens: Sequence[str], bos: bool = True, eos: bool = True) -> float:
        """Total log10 prob of the token sequence (KenLM convention: <s> unscored, </s> scored)."""
        ctx: Tuple[str, ...] = (BOS,) if bos else ()
        total = 0.0
        seq = list(tokens) + ([EOS] if eos else [])
        for w in seq:
            total += self.cond_logprob(ctx, w)
            ctx = (ctx + (w,))[-(self.order - 1):] if self.order > 1 else ()
        return total


# ------------------------------------------------------------------------------------- simulation

def load_phone_seqs(tphi_path: str, n_lines: int, min_len: int = 3,
                    keep: Sequence[str] = None) -> List[List[str]]:
    """Read up to n_lines of 𝒯_φ, drop OOV/[UNKNOWN] tokens, keep sequences with >= min_len phones."""
    keepset = set(keep) if keep is not None else None
    out: List[List[str]] = []
    op = gzip.open(tphi_path, "rt") if tphi_path.endswith(".gz") else open(tphi_path, "rt")
    with op as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            toks = [t for t in line.split() if keepset is None or t in keepset]
            if len(toks) >= min_len:
                out.append(toks)
    return out


def build_true_map(phone_freq: Dict[str, int], n_units: int, rng: np.random.Generator
                   ) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """Allocate n_units unit ids to phones ~ proportional to phone frequency (min 1 each, mirroring
    k-means allocating clusters by data mass). Returns (A_true: unit->phone, phone->units)."""
    phones = list(phone_freq)
    freqs = np.array([phone_freq[p] for p in phones], dtype=float)
    quota = np.maximum(1, np.round(freqs / freqs.sum() * n_units)).astype(int)
    # adjust to exactly n_units
    while quota.sum() > n_units:
        quota[np.argmax(quota)] -= 1
    while quota.sum() < n_units:
        quota[rng.integers(len(quota))] += 1
    a_true: Dict[int, str] = {}
    phone2units: Dict[str, List[int]] = {p: [] for p in phones}
    uid = 0
    for p, q in zip(phones, quota):
        for _ in range(int(q)):
            a_true[uid] = p
            phone2units[p].append(uid)
            uid += 1
    return a_true, phone2units


def simulate_units(phone_seqs: List[List[str]], phone2units: Dict[str, List[int]],
                   fertility: Dict[int, float], noise: float, rng: np.random.Generator
                   ) -> List[List[int]]:
    """Emit a unit stream per phone sequence: each phone -> n~fertility units, each a unit of that
    phone (w.p. 1-noise) or of a uniformly random phone (w.p. noise, emission error)."""
    all_units = [u for us in phone2units.values() for u in us]
    ferts = np.array(list(fertility.keys()))
    fert_p = np.array(list(fertility.values())); fert_p = fert_p / fert_p.sum()
    out: List[List[int]] = []
    for seq in phone_seqs:
        units: List[int] = []
        for ph in seq:
            n = int(rng.choice(ferts, p=fert_p))
            for _ in range(n):
                if rng.random() < noise or not phone2units.get(ph):
                    units.append(int(all_units[rng.integers(len(all_units))]))
                else:
                    pool = phone2units[ph]
                    units.append(int(pool[rng.integers(len(pool))]))
        if units:
            out.append(units)
    return out


# ------------------------------------------------------------------------------ counts + dedup

def dedup(seq: Sequence[int]) -> List[int]:
    out: List[int] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def skipk_bigrams(seqs: List[List[int]], k: int) -> Counter:
    """skip-(k-1) bigram counts: pairs (s[i], s[i+k])."""
    c: Counter = Counter()
    for s in seqs:
        for i in range(len(s) - k):
            c[(s[i], s[i + k])] += 1
    return c


def unit_4grams(seqs: List[List[int]]) -> Counter:
    """4-gram counts over BOS/EOS-padded deduped unit streams (boundary tokens as sentinels -1,-2)."""
    c: Counter = Counter()
    for s in seqs:
        p = [-1, -1, -1] + list(s) + [-2]  # BOS*3, EOS
        for i in range(3, len(p)):
            c[(p[i - 3], p[i - 2], p[i - 1], p[i])] += 1
    return c


# ------------------------------------------------------------------------------------- init/refine

def cdf_init(unit_freq: Dict[int, int], phone_freq: Dict[str, int]) -> Dict[int, str]:
    """A0(u) = phone at the same cumulative-frequency percentile as u (many-to-one CDF matching)."""
    units = sorted(unit_freq, key=lambda u: -unit_freq[u])
    utot = sum(unit_freq.values())
    phones = sorted(phone_freq, key=lambda p: -phone_freq[p])
    ptot = sum(phone_freq.values())
    # phone CDF upper edges
    edges, acc = [], 0.0
    for p in phones:
        acc += phone_freq[p] / ptot
        edges.append(acc)
    a: Dict[int, str] = {}
    cum = 0.0
    for u in units:
        mid = (cum + unit_freq[u] / utot / 2.0)
        cum += unit_freq[u] / utot
        j = 0
        while j < len(edges) - 1 and mid > edges[j]:
            j += 1
        a[u] = phones[j]
    return a


class _SkipModel:
    """Incremental induced phone skip-k bigram counts M and L1 cost to a fixed target T.

    Maintains M[i,j] = sum of unit-bigram counts whose endpoints currently map to phones i,j, and the
    L1 cost sum|M/total - T|. Reassigning one unit touches only O(deg) cells, so a move is evaluated
    and applied in O(deg), not O(all bigrams)."""

    def __init__(self, unit_bi: Counter, a: Dict[int, str], phones: List[str], target: np.ndarray):
        self.idx = {p: i for i, p in enumerate(phones)}
        self.T = target
        self.n = len(phones)
        self.out: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # u -> [(v, count)]
        self.inc: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # v -> [(u, count)]
        self.M = np.zeros((self.n, self.n))
        self.total = 0
        for (u, v), c in unit_bi.items():
            self.out[u].append((v, c)); self.inc[v].append((u, c)); self.total += c
            self.M[self.idx[a[u]], self.idx[a[v]]] += c
        self.total = max(self.total, 1)
        self.cost = float(np.abs(self.M / self.total - self.T).sum())

    def _deltas(self, u: int, a: Dict[int, str], p_new: int) -> Dict[Tuple[int, int], int]:
        p_old = self.idx[a[u]]
        dd: Dict[Tuple[int, int], int] = defaultdict(int)
        for v, c in self.out[u]:
            if v == u:
                dd[(p_old, p_old)] -= c; dd[(p_new, p_new)] += c
            else:
                jv = self.idx[a[v]]; dd[(p_old, jv)] -= c; dd[(p_new, jv)] += c
        for v, c in self.inc[u]:
            if v == u:
                continue  # self-loop already handled in out
            iv = self.idx[a[v]]; dd[(iv, p_old)] -= c; dd[(iv, p_new)] += c
        return dd

    def delta_cost(self, u: int, a: Dict[int, str], p_new: int) -> Tuple[float, Dict]:
        dd = self._deltas(u, a, p_new)
        dc = 0.0
        for (i, j), d in dd.items():
            old = abs(self.M[i, j] / self.total - self.T[i, j])
            new = abs((self.M[i, j] + d) / self.total - self.T[i, j])
            dc += new - old
        return dc, dd

    def apply(self, dd: Dict[Tuple[int, int], int]):
        for (i, j), d in dd.items():
            self.M[i, j] += d
        self.cost = float(np.abs(self.M / self.total - self.T).sum())


def bigram_refine(a: Dict[int, str], unit_bis: List[Counter], phone_targets: List[np.ndarray],
                  phones: List[str], max_sweeps: int = 8) -> Dict[int, str]:
    """Greedy coordinate descent minimizing summed-over-k L1 between induced and target phone skip-k
    bigram distributions; incremental per-move updates across the k in {1,2,3} models."""
    idx = {p: i for i, p in enumerate(phones)}
    a = dict(a)
    models = [_SkipModel(bi, a, phones, tgt) for bi, tgt in zip(unit_bis, phone_targets)]
    units = sorted({u for bi in unit_bis for pair in bi for u in pair})
    for _ in range(max_sweeps):
        changed = False
        for u in units:
            best_p, best_dc, best_dd = a[u], 0.0, None
            for p in phones:
                if p == a[u]:
                    continue
                pi = idx[p]
                dc = 0.0; dds = []
                for m in models:
                    d, dd = m.delta_cost(u, a, pi)
                    dc += d; dds.append(dd)
                if dc < best_dc - 1e-12:
                    best_dc, best_p, best_dd = dc, p, dds
            if best_dd is not None:
                for m, dd in zip(models, best_dd):
                    m.apply(dd)
                a[u] = best_p
                changed = True
        if not changed:
            break
    return a


# --------------------------------------------------------------------------------------------- ICM

def icm_polish(a: Dict[int, str], grams: Counter, lm: ArpaLM, phones: List[str],
               max_sweeps: int = 20) -> Dict[int, str]:
    """ICM maximizing L(A) = sum_g c(g) log p_4g(A(g)); only re-scores grams containing the moved
    unit. Boundary sentinels -1/-2 map to <s>/</s>."""
    a = dict(a)
    bnd = {-1: BOS, -2: EOS}

    def mp(x):  # map a unit id (or sentinel) to its phone symbol under current a
        return bnd[x] if x < 0 else a[x]

    # index: unit -> list of (gram, count)
    by_unit: Dict[int, List[Tuple[Tuple[int, int, int, int], int]]] = defaultdict(list)
    for g, c in grams.items():
        for u in set(g):
            if u >= 0:
                by_unit[u].append((g, c))

    units = sorted(by_unit)
    for _ in range(max_sweeps):
        changed = False
        for u in units:
            # collapse u's grams into distinct templates (u -> None; others -> current phone)
            templ: Dict[Tuple, int] = defaultdict(int)
            for g, c in by_unit[u]:
                templ[tuple(None if x == u else mp(x) for x in g)] += c
            best_p, best_s = a[u], None
            for p in phones:
                s = 0.0
                for t, c in templ.items():
                    s += c * lm.cond_logprob(
                        [p if x is None else x for x in t[:3]], p if t[3] is None else t[3])
                if best_s is None or s > best_s + 1e-9:
                    best_s, best_p = s, p
            if best_p != a[u]:
                a[u] = best_p
                changed = True
        if not changed:
            break
    return a


# ------------------------------------------------------ method (ii): HMM channel + Baum-Welch EM

def _phone_transition(lm: ArpaLM, phones: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed HMM transition T[i,j]=p(phone_j|phone_i), initial pi=p(phone|<s>), prior p(phone),
    all from the phoneme LM (bigram marginals), renormalized over the phone set."""
    n = len(phones)
    T = np.array([[10.0 ** lm.cond_logprob([phones[i]], phones[j]) for j in range(n)] for i in range(n)])
    T /= T.sum(1, keepdims=True)
    pi = np.array([10.0 ** lm.cond_logprob([BOS], p) for p in phones]); pi /= pi.sum()
    pr = np.array([10.0 ** lm.cond_logprob([], p) for p in phones]); pr /= pr.sum()
    return T, pi, pr


def hmm_decipher(unit_seqs: List[List[int]], phones: List[str], lm: ArpaLM, n_iters: int = 15,
                 init_a: Dict[int, str] = None, seed: int = 0, smooth: float = 1e-3
                 ) -> Tuple[Dict[int, str], float, np.ndarray]:
    """Ravi–Knight channel decipherment (method ii, bigram transition): hidden phone states with fixed
    LM transitions and a learned emission p(u|phi). Baum–Welch EM (scaled forward–backward). The
    emission simplex per phone is the anti-collapse the pure-LM hard map (i) lacks. Returns
    (A: unit->phone via argmax p(phi|u), final log-likelihood, emission E[phone,unit])."""
    n = len(phones)
    pidx = {p: i for i, p in enumerate(phones)}
    units = sorted({u for s in unit_seqs for u in s})
    uidx = {u: i for i, u in enumerate(units)}
    U = len(units)
    T, pi, prior = _phone_transition(lm, phones)
    rng = np.random.default_rng(seed)
    if init_a is None:
        E = rng.dirichlet(np.ones(U) * 0.5, size=n)
    else:
        E = np.full((n, U), 1e-3)
        for u, i in uidx.items():
            if u in init_a:
                E[pidx[init_a[u]], i] += 1.0
        E += rng.random((n, U)) * 1e-2
    E /= E.sum(1, keepdims=True)
    seqs = [[uidx[u] for u in s] for s in unit_seqs]

    ll = 0.0
    for _ in range(n_iters):
        ec = np.full((n, U), smooth)
        ll = 0.0
        for s in seqs:
            Tl = len(s)
            if Tl == 0:
                continue
            alpha = np.empty((Tl, n)); scale = np.empty(Tl)
            a = pi * E[:, s[0]]; scale[0] = a.sum() or 1e-300; alpha[0] = a / scale[0]
            for t in range(1, Tl):
                a = (alpha[t - 1] @ T) * E[:, s[t]]
                scale[t] = a.sum() or 1e-300; alpha[t] = a / scale[t]
            beta = np.empty((Tl, n)); beta[-1] = 1.0
            for t in range(Tl - 2, -1, -1):
                beta[t] = (T @ (E[:, s[t + 1]] * beta[t + 1])) / scale[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True) + 1e-300
            for t in range(Tl):
                ec[:, s[t]] += gamma[t]
            ll += float(np.log(scale).sum())
        E = ec / ec.sum(1, keepdims=True)

    post = E * prior[:, None]
    amap = {units[j]: phones[int(np.argmax(post[:, j]))] for j in range(U)}
    return amap, ll, E


# ----------------------------------------------------------------------------------------- metrics

def recovery(a: Dict[int, str], a_true: Dict[int, str], unit_freq: Dict[int, int]) -> Dict[str, float]:
    units = [u for u in a_true if u in a]
    corr = [1.0 if a[u] == a_true[u] else 0.0 for u in units]
    w = np.array([unit_freq.get(u, 0) for u in units], dtype=float)
    unweighted = float(np.mean(corr)) if corr else float("nan")
    weighted = float(np.average(corr, weights=w)) if w.sum() > 0 else unweighted
    return {"recovery_unweighted": unweighted, "recovery_weighted": weighted, "n_units": len(units)}


def decode_per(unit_seqs: List[List[int]], a: Dict[int, str], phone_seqs: List[List[str]]) -> float:
    """Map units->phones, run-length dedup both sides, mean Levenshtein PER."""
    from i6_experiments.users.wu.experiments.ssl.analysis.repr_audit import levenshtein

    tot_err = tot_len = 0
    for us, ref in zip(unit_seqs, phone_seqs):
        hyp = dedup([a[u] for u in us if u in a])
        ref_d = dedup(ref)
        tot_err += levenshtein(hyp, ref_d)
        tot_len += len(ref_d)
    return tot_err / max(tot_len, 1)


def unsup_metric(unit_seqs: List[List[int]], a: Dict[int, str], lm: ArpaLM, n_phones: int
                 ) -> Dict[str, float]:
    """§1.0 selection metric pieces: mean per-phone NLL under the LM and vocabulary usage U(P)."""
    used, total_lp, total_tok = set(), 0.0, 0
    for us in unit_seqs:
        hyp = dedup([a[u] for u in us if u in a])
        used.update(hyp)
        total_lp += lm.score_seq(hyp)
        total_tok += len(hyp) + 1  # + </s>
    nll_per_tok = -total_lp / max(total_tok, 1)  # log10 nats-free; monotone in NLL
    return {"nll_lm_per_tok_log10": nll_per_tok, "vocab_usage": len(used) / n_phones,
            "bits_per_phone": nll_per_tok / math.log10(2)}


# ------------------------------------------------------------------------------------------- driver

def run_once(tphi: str, arpa: str, n_units: int, n_lines: int, fertility: Dict[int, float],
             noise: float, seed: int, do_bigram: bool = True, do_icm: bool = True,
             do_hmm: bool = True, hmm_restarts: int = 4, icm_corpus_lines: int = 20000) -> Dict:
    from i6_experiments.users.wu.experiments.ssl.analysis.repr_audit import ARPABET_39

    rng = np.random.default_rng(seed)
    lm = ArpaLM(arpa)
    phone_seqs = load_phone_seqs(tphi, n_lines, keep=ARPABET_39)
    phone_freq = Counter(p for s in phone_seqs for p in s)
    phones = [p for p in ARPABET_39 if phone_freq[p] > 0]
    a_true, phone2units = build_true_map(phone_freq, n_units, rng)

    unit_seqs_raw = simulate_units(phone_seqs, phone2units, fertility, noise, rng)
    unit_seqs = [dedup(s) for s in unit_seqs_raw]  # VAD/dedup analogue
    unit_freq = Counter(u for s in unit_seqs for u in s)

    # target phone skip-k bigram distributions from the same 𝒯_φ sample (available text stats)
    phone_ids = {p: i for i, p in enumerate(phones)}
    phone_int_seqs = [[phone_ids[p] for p in s if p in phone_ids] for s in phone_seqs]

    def phone_target(k):
        M = np.zeros((len(phones), len(phones)))
        for s in phone_int_seqs:
            for i in range(len(s) - k):
                M[s[i], s[i + k]] += 1
        return M / max(M.sum(), 1)

    a = cdf_init(unit_freq, phone_freq)
    stages = {"cdf": dict(a)}

    if do_bigram:
        unit_bis = [skipk_bigrams(unit_seqs, k) for k in (1, 2, 3)]
        targets = [phone_target(k) for k in (1, 2, 3)]
        a = bigram_refine(a, unit_bis, targets, phones)
        stages["bigram"] = dict(a)

    if do_icm:
        grams = unit_4grams(unit_seqs[:icm_corpus_lines])
        a = icm_polish(dict(stages.get("bigram", stages["cdf"])), grams, lm, phones)
        stages["icm"] = dict(a)

    if do_hmm:
        warm = stages.get("bigram", stages["cdf"])
        best_ll, best_a = None, None
        for r in range(hmm_restarts):
            init = warm if r == 0 else None  # r0 warm-start, rest random restarts
            amap, ll, _ = hmm_decipher(unit_seqs, phones, lm, init_a=init, seed=seed * 100 + r)
            if best_ll is None or ll > best_ll:
                best_ll, best_a = ll, amap
        stages["hmm"] = best_a
        result_hmm_ll = best_ll

    result = {"n_units": n_units, "n_phones": len(phones), "noise": noise, "seed": seed,
              "fertility": fertility, "n_unit_seqs": len(unit_seqs),
              "dedup_ratio": sum(len(s) for s in unit_seqs) / max(sum(len(dedup(s)) for s in phone_int_seqs), 1)}
    for name, aa in stages.items():
        rec = recovery(aa, a_true, unit_freq)
        result[name] = {**rec, "per": decode_per(unit_seqs, aa, phone_seqs),
                        **unsup_metric(unit_seqs, aa, lm, len(phones))}
    return result


def _fmt(r: Dict) -> str:
    lines = [f"units={r['n_units']} phones={r['n_phones']} noise={r['noise']} "
             f"fert={r['fertility']} dedupρ={r['dedup_ratio']:.2f} seed={r['seed']}"]
    for st in ("cdf", "bigram", "icm", "hmm"):
        if st in r:
            s = r[st]
            lines.append(f"  {st:7s} recov(w)={s['recovery_weighted']:.3f} "
                         f"recov(u)={s['recovery_unweighted']:.3f} PER={s['per']:.3f} "
                         f"U={s['vocab_usage']:.2f} bits/ph={s['bits_per_phone']:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SAE §1a decipherment (i) on simulated unit streams")
    ap.add_argument("--tphi", default="output/sae/0b/tphi.txt.gz")
    ap.add_argument("--arpa", default="output/sae/1a/phoneme_lm_o4.arpa.gz")
    ap.add_argument("--n_units", type=int, default=100)
    ap.add_argument("--n_lines", type=int, default=20000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--fertility", default="1:1.0")  # e.g. "1:0.7,2:0.25,3:0.05"
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bigram", action="store_true")
    ap.add_argument("--no-icm", action="store_true")
    ap.add_argument("--no-hmm", action="store_true")
    ap.add_argument("--hmm-restarts", type=int, default=4)
    args = ap.parse_args()
    fert = {int(kv.split(":")[0]): float(kv.split(":")[1]) for kv in args.fertility.split(",")}
    r = run_once(args.tphi, args.arpa, args.n_units, args.n_lines, fert, args.noise, args.seed,
                 do_bigram=not args.no_bigram, do_icm=not args.no_icm, do_hmm=not args.no_hmm,
                 hmm_restarts=args.hmm_restarts)
    print(_fmt(r))
