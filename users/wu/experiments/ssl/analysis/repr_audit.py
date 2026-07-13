"""SAE Phase 0a — representation-audit metrics (evaluation-only, against MFA gold).

Given a frozen (layer, K) unit stream ``c`` (per-frame k-means ids at 25 Hz) and the MFA gold
per-frame phone labels ``g`` on the same 25 Hz grid, compute the numbers SAE_PLAN §0a exit-gate
records: cluster ``purity``, ``PNMI`` = I(k;phi)/H(phi), conditional entropy H(phi|u), the
hard-assignment ``oracle-map PER`` (the hard ceiling of §1a(i)), and the dedup length ratio rho.

The metric functions depend on numpy only (no pandas / sisyphus) so they are unit-testable
standalone; the gold loader (``load_gold_phonemes`` / ``frame_phone_labels``) reads the cached
gilkeyio ``librispeech-alignments`` parquet, mirroring analysis/seg_diag.py:_load_gold but keeping
phone *identity* (which seg_diag discards).

Conventions (documented in SAE_0.md):
- Phones are collapsed to the stress-free 39-phone CMU/ARPAbet set (trailing stress digits stripped).
- Non-speech MFA labels {sil,sp,spn,"",noise,nsn,lau,<unk>,<eps>} and uncovered gaps -> ``SIL``.
- purity / PNMI / H(phi|u) are frame-level and, by default, INCLUDE the SIL class (units model
  silence too). oracle-map PER is token-level over run-length-deduped sequences with SIL dropped
  (PER is scored over spoken phones only) -- both behaviours are parameterised.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Stress-free CMU/ARPAbet monophone set (39) + silence class. Fixed order -> stable ids across splits.
ARPABET_39 = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
SIL = "SIL"
PHONE2ID: Dict[str, int] = {p: i for i, p in enumerate(ARPABET_39)}
PHONE2ID[SIL] = len(ARPABET_39)
SIL_ID = PHONE2ID[SIL]
NUM_PHONES = len(PHONE2ID)  # 40 incl. SIL

_NONSPEECH = {"sil", "sp", "spn", "", "<unk>", "<eps>", "noise", "nsn", "lau"}


def canonical_phone(label: str) -> str:
    """MFA phone label -> stress-free class. Non-speech -> SIL; strip trailing stress digits."""
    s = str(label)
    if s.lower() in _NONSPEECH:
        return SIL
    s = s.rstrip("0123456789")  # AA0/AA1/AA2 -> AA
    return s if s in PHONE2ID else SIL


def frame_phone_labels(phonemes: Sequence[dict], num_frames: int, frame_rate_hz: float = 25.0) -> np.ndarray:
    """Build a length-``num_frames`` int array of per-frame phone ids from an MFA ``phonemes`` list
    (entries ``{start,end,phoneme}`` in seconds). Frame ``t`` takes its center time ``(t+0.5)/fps``;
    frames not covered by any speech interval are SIL. Overlaps resolved by first-covering interval.
    """
    g = np.full(int(num_frames), SIL_ID, dtype=np.int64)
    if num_frames <= 0:
        return g
    centers = (np.arange(num_frames) + 0.5) / frame_rate_hz
    for p in phonemes:
        pid = PHONE2ID[canonical_phone(p["phoneme"])]
        if pid == SIL_ID:
            continue  # leave as SIL background; only paint speech intervals
        lo = int(np.searchsorted(centers, float(p["start"]), side="left"))
        hi = int(np.searchsorted(centers, float(p["end"]), side="left"))
        if hi > lo:
            g[lo:hi] = pid
    return g


def load_gold_phonemes(splits: Sequence[str], hf_home: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load {utt_id -> phonemes-array} from the cached gilkeyio alignments for the given HF splits
    (e.g. 'validation.other'). Mirrors seg_diag.py:_load_gold."""
    import pandas as pd

    split_map = {
        "validation.clean": "dev_clean", "validation.other": "dev_other",
        "test.clean": "test_clean", "test.other": "test_other",
        "train.clean.100": "train_clean_100", "train.clean.360": "train_clean_360",
        "train.other.500": "train_other_500",
    }
    hf = hf_home or os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")
    base = os.path.join(hf, "hub", "datasets--gilkeyio--librispeech-alignments", "snapshots")
    gold: Dict[str, np.ndarray] = {}
    for src in splits:
        name = split_map[src]
        files = sorted(glob.glob(os.path.join(base, "*", "data", f"{name}-*.parquet")))
        assert files, f"no gold alignment parquet for split {src!r} ({name}) under {base}"
        for fp in files:
            df = pd.read_parquet(fp, columns=["id", "phonemes"])
            for r in df.itertuples():
                gold[r.id] = r.phonemes
    return gold


# ---------------------------------------------------------------------------------------------
# Metric primitives (numpy-only, frame-aligned c[T] units vs g[T] gold phone ids)
# ---------------------------------------------------------------------------------------------

def confusion(c: np.ndarray, g: np.ndarray, num_units: int, num_phones: int = NUM_PHONES) -> np.ndarray:
    """Co-occurrence count matrix N[num_units, num_phones] over frame-aligned c,g."""
    assert c.shape == g.shape, (c.shape, g.shape)
    N = np.zeros((num_units, num_phones), dtype=np.float64)
    np.add.at(N, (c.astype(np.int64), g.astype(np.int64)), 1.0)
    return N


def purity(N: np.ndarray) -> float:
    """Cluster purity = sum_k max_phi N(k,phi) / sum N."""
    tot = N.sum()
    return float(N.max(axis=1).sum() / tot) if tot > 0 else 0.0


def pnmi(N: np.ndarray) -> float:
    """Phone-normalized MI = I(k;phi) / H(phi)."""
    tot = N.sum()
    if tot <= 0:
        return 0.0
    Pk = N.sum(axis=1) / tot
    Pphi = N.sum(axis=0) / tot
    Pkphi = N / tot
    nz = Pkphi > 0
    mi = float((Pkphi[nz] * np.log(Pkphi[nz] / (Pk[:, None] * Pphi[None, :])[nz])).sum())
    Hphi = float(-(Pphi[Pphi > 0] * np.log(Pphi[Pphi > 0])).sum())
    return mi / Hphi if Hphi > 0 else 0.0


def cond_entropy_phi_given_u(N: np.ndarray) -> float:
    """H(phi|u) in nats -- many-to-many-ness; predicts §1a(ii) fertility-HMM headroom."""
    tot = N.sum()
    if tot <= 0:
        return 0.0
    Pk = N.sum(axis=1) / tot
    H = 0.0
    for k in range(N.shape[0]):
        row = N[k]
        s = row.sum()
        if s <= 0:
            continue
        p = row[row > 0] / s
        H += Pk[k] * float(-(p * np.log(p)).sum())
    return H


def run_length_dedup(seq: np.ndarray) -> np.ndarray:
    """Collapse consecutive-equal runs: [a,a,b,b,b,a] -> [a,b,a]."""
    seq = np.asarray(seq)
    if seq.size == 0:
        return seq
    keep = np.ones(seq.size, dtype=bool)
    keep[1:] = seq[1:] != seq[:-1]
    return seq[keep]


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """Edit distance (sub/ins/del cost 1) between two int sequences."""
    a, b = list(a), list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)]


def oracle_map(N: np.ndarray, sil_id: int = SIL_ID) -> np.ndarray:
    """A*(u) = argmax_phi N(u,phi). Empty units (never seen) map to SIL."""
    A = N.argmax(axis=1)
    A[N.sum(axis=1) == 0] = sil_id
    return A.astype(np.int64)


def gold_phone_tokens(g: np.ndarray, drop_sil: bool = True, sil_id: int = SIL_ID) -> np.ndarray:
    """Frame gold ids -> deduped phone token sequence (SIL dropped by default)."""
    tok = run_length_dedup(g)
    return tok[tok != sil_id] if drop_sil else tok


def oracle_map_per(
    units_by_utt: Dict[str, np.ndarray],
    gold_by_utt: Dict[str, np.ndarray],
    num_units: int,
    sil_id: int = SIL_ID,
) -> Tuple[float, np.ndarray]:
    """Hard-assignment oracle-map PER (SAE_PLAN §0a): build A* from ALL frames, then per-utt map
    frame units -> phones, dedup, drop SIL, and score PER = sum edits / sum gold-phone tokens
    against the deduped gold phone sequence. Returns (PER, A*)."""
    keys = [k for k in units_by_utt if k in gold_by_utt]
    N = np.zeros((num_units, NUM_PHONES), dtype=np.float64)
    for k in keys:
        c, g = units_by_utt[k], gold_by_utt[k]
        n = min(c.size, g.size)
        np.add.at(N, (c[:n].astype(np.int64), g[:n].astype(np.int64)), 1.0)
    A = oracle_map(N, sil_id=sil_id)
    tot_err, tot_len = 0, 0
    for k in keys:
        c = units_by_utt[k].astype(np.int64)
        pred = run_length_dedup(A[c])
        pred = pred[pred != sil_id]
        ref = gold_phone_tokens(gold_by_utt[k], drop_sil=True, sil_id=sil_id)
        tot_err += levenshtein(pred, ref)
        tot_len += ref.size
    per = float(tot_err / tot_len) if tot_len > 0 else float("nan")
    return per, A


def dedup_length_ratio(
    units_by_utt: Dict[str, np.ndarray],
    gold_by_utt: Dict[str, np.ndarray],
    sil_id: int = SIL_ID,
) -> float:
    """rho = E[|u|] / E[|phi|] over utterances: deduped unit tokens vs deduped gold phone tokens
    (SIL dropped from both)."""
    keys = [k for k in units_by_utt if k in gold_by_utt]
    if not keys:
        return float("nan")
    u_lens, phi_lens = [], []
    for k in keys:
        u = run_length_dedup(units_by_utt[k])
        u_lens.append(u.size)
        phi_lens.append(gold_phone_tokens(gold_by_utt[k], drop_sil=True, sil_id=sil_id).size)
    mphi = float(np.mean(phi_lens))
    return float(np.mean(u_lens) / mphi) if mphi > 0 else float("nan")


def audit_layer_k(
    units_by_utt: Dict[str, np.ndarray],
    gold_by_utt: Dict[str, np.ndarray],
    num_units: int,
    include_sil_in_purity: bool = True,
) -> Dict[str, float]:
    """One (layer,K) row of the §0a table over frame-aligned dev units/gold."""
    keys = [k for k in units_by_utt if k in gold_by_utt]
    cs, gs = [], []
    for k in keys:
        c, g = units_by_utt[k], gold_by_utt[k]
        n = min(c.size, g.size)
        cs.append(c[:n]); gs.append(g[:n])
    C = np.concatenate(cs); G = np.concatenate(gs)
    if not include_sil_in_purity:
        m = G != SIL_ID
        C, G = C[m], G[m]
    N = confusion(C, G, num_units)
    per, _ = oracle_map_per(units_by_utt, gold_by_utt, num_units)
    return {
        "purity": purity(N),
        "pnmi": pnmi(N),
        "H_phi_given_u": cond_entropy_phi_given_u(N),
        "oracle_map_per": per,
        "rho": dedup_length_ratio(units_by_utt, gold_by_utt),
        "num_frames": int(N.sum()),
        "num_utts": len(keys),
    }
