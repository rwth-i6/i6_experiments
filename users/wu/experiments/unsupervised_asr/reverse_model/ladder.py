"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_ladder_jobs.py (the string jobs) and
``prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_lexlat_v2_ladder_v1.py``
(the phi fits and the theta-at-cold-init arm configs), with ``config_sae_4a_supervised_k2_diag_v1.
_schedules`` and ``_model_args_delta`` (the schedules and the k2 block every ladder arm copies).

ANALYSIS ONLY (uses transcripts).  L2-0 of SAE_4A_lexlat_v2.md, the reverse model's competence
ladder: a DISCLOSED LABEL-USING DIAGNOSTIC.  It is built from the 10 h seed's gold phone strings;
nothing here enters a gate, selects a checkpoint or initialises a main-line arm.

phi_rho (:func:`corrupted_phi`), per rho in :data:`RHOS`:
  1. :class:`CorruptSeedGoldJob` on the seed gold json (``data.gold.SeedGoldPhonesJob``'s
     ``seed_gold_phones.json``), seed :data:`CORRUPTION_SEED`: exactly ``round(rho n_u)`` tokens of
     each utterance substituted by a draw from the seed's gold unigram renormalised without the
     original symbol; length kept; no run-collapse (the seed gold strings carry adjacent repeats, so
     that is the convention the targets and the loader already satisfy).
  2. ``data.gold.PhoneTargetHdfJob`` on the corrupted json with the gold targets' other arguments
     (``ids_json`` = the seed ids, ``label_key`` = None).
  3. The gold phi's fit (``supervised.blankfree_supervised_reverse_init``) with the corrupted json as
     ``gold_json`` and the corrupted targets as ``targets_hdf`` (the label source is the only delta
     from the gold phi); checkpoint = the fixed final epoch 8, as the gold phi's.

permphi (:func:`permuted_phi`): the same fit on the seed gold strings under one fixed random
permutation of the 39 phones, seed :data:`PERM_SEED` (:class:`PermuteSeedGoldJob`).

THE CORRUPTION (:func:`corrupt_string`), per utterance u with n_u tokens:

* exactly ``round(rho * n_u)`` positions are substituted (Python's built-in ``round``, i.e. half to
  even); the positions are the first ``round(rho * n_u)`` entries of a uniform random permutation of
  ``range(n_u)``;
* each substituted token is replaced by a symbol drawn from the seed's gold unigram (counts over ALL
  seed gold tokens, train and held) renormalised without the original symbol, one uniform
  ``rng.random()`` per substituted position, in permutation order, inverted through that
  renormalised CDF;
* the rng is ``numpy.random.default_rng(SeedSequence([seed, tag_key(tag)]))``, ``tag_key`` the first
  8 bytes of sha256(tag) (:func:`utterance_rng`): it depends on the seed and the tag only, so the draw
  is independent of utterance order AND of rho, and the ladder is nested (the substitutions at a
  smaller rho are the first ones made at a larger rho, with the same replacement symbols);
* the length is kept; nothing is collapsed (the collapse count is 0 by construction; the job reports
  the adjacent-repeat count of the gold and of the corrupted strings beside it).

SYMBOL SET.  The seed gold strings hold exactly the 39 ARPAbet phones of ``phones.PHONES`` (no SIL, no
other token).  The job asserts that.

REPORTED (``corruption.json`` / ``corruption.txt``), per part all / train / held (the seed split's
2821 / 28 manifests): the realised substitution rate, the PER of the corrupted strings against the
gold with :func:`edit_counts` ((S + D + I) / N; the Levenshtein alignment can be cheaper than the
positional substitutions, so PER <= the realised rate), the collapse count, the adjacent-repeat
counts.  Descriptive only.

THE THETA-AT-COLD-INIT ARMS (:func:`rt_train_config`, nodes R1 / R2 of the source, :data:`RT_PACKS`):
every arm is D10e's ``supphi_k2lat`` (the bed; tau held at 2.0 and the phase's N = 20 learning rates
truncated to :data:`NUM_SUBEPOCHS` = 8, :func:`schedules`; the in-house word graph's k2 block at rung
1000, on-set 1, ramp 3, lam 1; both models trainable; kept 1 / 2 / 4 / 8) with ONLY these deltas:

* theta at the cold flat init ``FlatRecognizerInitJob(net_args=NET_ARGS, seed=0)`` (the data dict's
  ``flat_checkpoint`` is replaced by it, as the source pinned it);
* ``reverse_checkpoint_path`` per arm (none for ``cold_ctl``: phi at the bed's random init);
* ``lexlat_k2_chunk_seqs`` = :data:`RT_K2_CHUNK_SEQS` (a launch granularity: k2 prunes per sequence,
  so no score or gradient moves);
* ``rt_r0_s2`` only: the campaign's second seed :data:`SECOND_SEED` (flat seed 1, RETURNN
  ``random_seed`` 1, the train dataset's ``random_seed_offset`` 1000).

Registered reads for R1 / R2 (read by hand; no job here applies them): dev-other greedy PER per arm at
ep1 / 2 / 4 / 8; at ep8 LIFT = PER < :data:`LIFT_PER`, PARTIAL = PER < :data:`PARTIAL_PER`, NO LIFT
otherwise.

Port changes:

* Inputs: the seed inputs (gold json, ids, split manifests), the bed's data dict and the word-graph
  dict are the caller's; the source pinned banked paths.  The gold phi fit is
  ``supervised.blankfree_supervised_reverse_init`` (a RETURNN training, see its module docstring) in
  place of ``BlankfreeSupervisedReverseInitJob``.
* The R packs run one ``training.jobs.train_arm`` job per arm (:func:`build_rt`) instead of two 4-GPU
  ``PackedBlankfreeTrainJob`` nodes; the per-arm allocation stays the pack's 11.5 h
  (``training.jobs.TIME_RQMT``).
* Cut: the k2 pre-flight (``LadderK2PreflightJob``, ``parse_step_line``, ``render_preflight``), node P
  (the corrupted-phi arms on p0's theta), the per-arm PER reads and paired deltas, statistic (c)
  (``decode_gap``, ``REUSED_GAP_READS``), D14's decphi, statistic (b) of the competence reads (not
  ported in ``genmarg``) and the dev-other competence reads.
* ``edit_counts`` is a verbatim local copy of ``sae/emc/eval_jobs.edit_counts``.
* The builders return their jobs and register no outputs; the source config's ``sys.path`` prolog is
  dropped.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sisyphus import Job, Task, tk

__all__ = ["CorruptSeedGoldJob", "corrupt_string", "corrupt_all", "gold_unigram", "utterance_rng",
           "corruption_stats", "edit_counts", "SEED_UTTERANCES", "PermuteSeedGoldJob",
           "phone_permutation", "permute_all", "RHOS", "CORRUPTION_SEED", "PERM_SEED", "tag",
           "NUM_SUBEPOCHS", "KEEP_EPOCHS", "RT_K2_CHUNK_SEQS", "SECOND_SEED", "RT_PACKS", "LIFT_PER",
           "PARTIAL_PER", "schedules", "corrupted_phi", "permuted_phi", "ladder_phis",
           "competence_reads", "rt_model_args_delta", "rt_train_config", "build_fits", "build_rt"]

#: the seed census every consumer asserts (2849, split 2821 / 28)
SEED_UTTERANCES = 2849


def _phones39() -> List[str]:
    from ..phones import PHONES

    assert PHONES[-1] == "SIL" and len(PHONES) == 40
    return list(PHONES[:-1])


def tag_key(tag: str) -> int:
    """A stable 64-bit integer of the tag (sha256, first 8 bytes big-endian); no Python ``hash``."""
    return int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big")


def utterance_rng(seed: int, tag: str) -> np.random.Generator:
    """The utterance's generator: a function of ``seed`` and ``tag`` only (module doc)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), tag_key(tag)]))


def gold_unigram(gold: Dict[str, Sequence[str]]) -> np.ndarray:
    """``[39]`` relative frequencies of the 39 phones over ALL seed gold tokens (``_phones39`` order)."""
    phones = _phones39()
    counts = Counter(p for seq in gold.values() for p in seq)
    unknown = sorted(set(counts) - set(phones))
    assert not unknown, f"seed gold tokens outside the 39 phones: {unknown}"
    c = np.array([counts[p] for p in phones], dtype=np.float64)
    return c / c.sum()


def _renormalised_cdfs(unigram: np.ndarray) -> np.ndarray:
    """``[39, 39]``: row k = the CDF of the unigram with symbol k removed and the rest renormalised."""
    n = len(unigram)
    cdfs = np.empty((n, n), dtype=np.float64)
    for k in range(n):
        p = unigram.copy()
        p[k] = 0.0
        assert p.sum() > 0
        cdfs[k] = np.cumsum(p / p.sum())
    return cdfs


def corrupt_string(seq: Sequence[str], rho: float, rng: np.random.Generator,
                   cdfs: np.ndarray) -> Tuple[List[str], int]:
    """``(corrupted, n_substituted)`` of one utterance (module doc); length kept, no collapse."""
    phones = _phones39()
    index = {p: i for i, p in enumerate(phones)}
    n = len(seq)
    k = int(round(rho * n))
    assert 0 <= k <= n, (rho, n, k)
    order = rng.permutation(n)
    u = rng.random(k)
    out = list(seq)
    for position, draw in zip(order[:k], u):
        original = index[seq[position]]
        row = cdfs[original]
        j = int(np.searchsorted(row, draw, side="right"))
        j = min(j, len(row) - 1)
        while row[j] - (row[j - 1] if j else 0.0) <= 0.0:  # a zero-mass slot (only the original)
            j = j - 1 if j == len(row) - 1 else j + 1
        assert j != original
        out[int(position)] = phones[j]
    return out, k


def corrupt_all(gold: Dict[str, Sequence[str]], rho: float, seed: int) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """``({tag: corrupted}, {tag: n_substituted})`` in the gold json's key order."""
    cdfs = _renormalised_cdfs(gold_unigram(gold))
    out, subs = {}, {}
    for tag, seq in gold.items():
        out[tag], subs[tag] = corrupt_string(seq, rho, utterance_rng(seed, tag), cdfs)
    return out, subs


def _repeats(seq: Sequence[str]) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if a == b)


def edit_counts(hyp: Sequence[str], ref: Sequence[str]) -> Tuple[int, int, int]:
    """Levenshtein backtrace -> (substitutions, deletions, insertions) of ``hyp`` against ``ref``.

    Deletion = a reference symbol missing from the hypothesis, insertion = a hypothesis symbol with
    no reference counterpart (the sclite convention).  Unit costs; ties resolve substitution before
    deletion before insertion, which changes the S/D/I split of an ambiguous alignment but never
    their sum (and PER is read off the sum).
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = distance between ref[:i] and hyp[:j]
    prev = list(range(m + 1))
    ops = [[0] * (m + 1) for _ in range(n + 1)]  # 0 = match, 1 = sub, 2 = del, 3 = ins
    for j in range(1, m + 1):
        ops[0][j] = 3
    for i in range(1, n + 1):
        cur = [prev[0] + 1] + [0] * m
        ops[i][0] = 2
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            c_sub = prev[j - 1] + (0 if same else 1)
            c_del = prev[j] + 1
            c_ins = cur[j - 1] + 1
            best = min(c_sub, c_del, c_ins)
            cur[j] = best
            ops[i][j] = 0 if (same and best == c_sub) else (1 if best == c_sub else (2 if best == c_del else 3))
        prev = cur
    i, j, sub, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        op = ops[i][j]
        if i > 0 and j > 0 and op in (0, 1):
            sub += op == 1
            i, j = i - 1, j - 1
        elif i > 0 and op == 2:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def corruption_stats(gold, corrupted, subs, parts: Dict[str, List[str]]) -> dict:
    """Per part: realised rate, PER (:func:`edit_counts`), collapses, adjacent repeats."""
    record = {}
    for part, tags in parts.items():
        s = d = i = n = n_sub = changed = 0
        for tag in tags:
            a, b, c = edit_counts(corrupted[tag], gold[tag])
            s += a; d += b; i += c; n += len(gold[tag])
            n_sub += subs[tag]
            changed += sum(x != y for x, y in zip(corrupted[tag], gold[tag]))
        record[part] = {
            "utterances": len(tags), "gold_tokens": n, "corrupted_tokens": sum(len(corrupted[t]) for t in tags),
            "substituted": n_sub, "realised_substitution_rate": n_sub / n,
            "positions_differing": changed,
            "per": (s + d + i) / n, "sub": s, "del": d, "ins": i,
            "collapses": 0,
            "adjacent_repeats_gold": sum(_repeats(gold[t]) for t in tags),
            "adjacent_repeats_corrupted": sum(_repeats(corrupted[t]) for t in tags),
        }
    return record


class CorruptSeedGoldJob(Job):
    """ANALYSIS ONLY (uses transcripts).  The seed gold strings with a fraction ``rho`` of tokens
    substituted (module doc).

    :param gold_json: ``SeedGoldPhonesJob``'s ``seed_gold_phones.json`` (``{tag: [phone, ...]}``), the
        gold the gold phi fits on.
    :param train_segments / cv_segments: the seed split's manifests, for the per-part report only.
    :param rho: the substitution fraction.
    :param seed: the corruption seed (0, the phase's).

    Outputs: ``corrupted_phones.json`` (the gold json's format and key order), ``corruption.json``,
    ``corruption.txt``.
    """

    def __init__(self, *, gold_json: tk.Path, train_segments: tk.Path, cv_segments: tk.Path,
                 rho: float, seed: int = 0):
        super().__init__()
        assert 0.0 < float(rho) <= 1.0, rho
        self.gold_json = gold_json
        self.train_segments = train_segments
        self.cv_segments = cv_segments
        self.rho = float(rho)
        self.seed = int(seed)
        self.out_phones = self.output_path("corrupted_phones.json")
        self.out_stats = self.output_path("corruption.json")
        self.out_report = self.output_path("corruption.txt")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        gold = json.loads(Path(self.gold_json.get_path()).read_text())
        parts = {"train": Path(self.train_segments.get_path()).read_text().splitlines(),
                 "held": Path(self.cv_segments.get_path()).read_text().splitlines()}
        assert len(gold) == SEED_UTTERANCES and set(gold) == set(parts["train"]) | set(parts["held"])
        assert not set(parts["train"]) & set(parts["held"])
        assert all(gold[t] for t in gold), "empty seed gold string"
        parts = {"all": list(gold), **parts}
        corrupted, subs = corrupt_all(gold, self.rho, self.seed)
        assert all(len(corrupted[t]) == len(gold[t]) for t in gold)
        record = corruption_stats(gold, corrupted, subs, parts)
        record = {"rho": self.rho, "seed": self.seed, "rounding": "python round (half to even)",
                  "collapse_convention": "none (the seed gold strings carry adjacent repeats)",
                  "parts": record}
        with open(self.out_phones.get_path(), "w") as fh:
            json.dump(corrupted, fh)
        Path(self.out_stats.get_path()).write_text(json.dumps(record, indent=2) + "\n")
        lines = [f"rho={self.rho} seed={self.seed} {part}: realised_rate={r['realised_substitution_rate']:.6f} "
                 f"PER={r['per']:.6f} (S={r['sub']} D={r['del']} I={r['ins']} N={r['gold_tokens']}) "
                 f"collapses={r['collapses']} repeats gold/corrupted="
                 f"{r['adjacent_repeats_gold']}/{r['adjacent_repeats_corrupted']} utts={r['utterances']}"
                 for part, r in record["parts"].items()]
        Path(self.out_report.get_path()).write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def phone_permutation(seed: int) -> Dict[str, str]:
    """The fixed bijection of the 39 phones: ``phones[i] -> phones[default_rng(seed).permutation(39)[i]]``."""
    phones = _phones39()
    perm = np.random.default_rng(int(seed)).permutation(len(phones))
    mapping = {phones[i]: phones[int(j)] for i, j in enumerate(perm)}
    assert sorted(mapping.values()) == sorted(phones)
    return mapping


def permute_all(gold: Dict[str, Sequence[str]], mapping: Dict[str, str]) -> Tuple[Dict[str, List[str]], int]:
    """``({tag: permuted}, n_unmapped)``: every token through ``mapping``; others (SIL, ...) unchanged."""
    out, unmapped = {}, 0
    for tag, seq in gold.items():
        out[tag] = [mapping.get(p, p) for p in seq]
        unmapped += sum(1 for p in seq if p not in mapping)
    return out, unmapped


class PermuteSeedGoldJob(Job):
    """ANALYSIS ONLY (uses transcripts).  The seed gold strings under one fixed random permutation of
    the 39 phones (module doc, permphi).

    :param gold_json: ``SeedGoldPhonesJob``'s ``seed_gold_phones.json``, the gold the gold phi fits on.
    :param seed: the permutation seed (0, the dispatch's).

    Outputs: ``permuted_phones.json`` (the gold json's format and key order), ``permutation.json``
    (the map, its fixed points, token counts), ``permutation.txt``.
    """

    def __init__(self, *, gold_json: tk.Path, seed: int = 0):
        super().__init__()
        self.gold_json = gold_json
        self.seed = int(seed)
        self.out_phones = self.output_path("permuted_phones.json")
        self.out_stats = self.output_path("permutation.json")
        self.out_report = self.output_path("permutation.txt")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        gold = json.loads(Path(self.gold_json.get_path()).read_text())
        assert len(gold) == SEED_UTTERANCES and all(gold[t] for t in gold)
        mapping = phone_permutation(self.seed)
        permuted, unmapped = permute_all(gold, mapping)
        assert all(len(permuted[t]) == len(gold[t]) for t in gold)
        inverse = {v: k for k, v in mapping.items()}
        assert all([inverse.get(p, p) for p in permuted[t]] == list(gold[t]) for t in gold)
        fixed = sorted(p for p, q in mapping.items() if p == q)
        n = sum(len(s) for s in gold.values())
        changed = sum(a != b for t in gold for a, b in zip(permuted[t], gold[t]))
        record = {"seed": self.seed, "rule": "numpy default_rng(seed).permutation(39) over prior.PHONES[:39]",
                  "mapping": mapping, "fixed_points": fixed, "tokens": n, "tokens_changed": changed,
                  "tokens_unmapped_kept": unmapped, "utterances": len(gold),
                  "adjacent_repeats_gold": sum(_repeats(s) for s in gold.values()),
                  "adjacent_repeats_permuted": sum(_repeats(s) for s in permuted.values())}
        with open(self.out_phones.get_path(), "w") as fh:
            json.dump(permuted, fh)
        Path(self.out_stats.get_path()).write_text(json.dumps(record, indent=2) + "\n")
        lines = [f"permutation seed={self.seed} ({record['rule']}):"]
        lines += [f"  {p} -> {q}" for p, q in mapping.items()]
        lines.append(f"fixed points {len(fixed)}: {fixed}; tokens {n}, changed {changed}, unmapped kept "
                     f"{unmapped}; repeats gold/permuted {record['adjacent_repeats_gold']}/"
                     f"{record['adjacent_repeats_permuted']}; utts {len(gold)}")
        Path(self.out_report.get_path()).write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


# -- the ladder's constants (config_sae_4a_lexlat_v2_ladder_v1) ---------------------------------
ALIAS = "sae/4a/lexlat_v2_ladder"

#: the ladder (L2-0 "Inputs") and the corruption seed
RHOS: Tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)
CORRUPTION_SEED = 0
#: permphi's permutation seed (dispatch: seed 0)
PERM_SEED = 0
#: D10e's sub-epochs and kept checkpoints (``config_sae_4a_supervised_k2_diag_v1``)
NUM_SUBEPOCHS = 8
KEEP_EPOCHS: Tuple[int, ...] = (1, 2, 4, 8)
#: the k2 block of D10e's ``sup_k2lat``: rung 1000, on-set 1 (ramp, lam, beams, min-active at the
#: node-1 constants, ``lexlat_k2_model_args``' defaults)
K2_MAX_ACTIVE = 1000
K2_ONSET = 1
#: sequences per k2 pruned-intersection call in the R arms' training step (a launch granularity, not
#: an arm constant: 16 per call overflowed k2's int32 window sum on the flat theta, 1-4 are safe)
RT_K2_CHUNK_SEQS = 4
#: the campaign's registered second seed (``config_sae_4a_prepro_pack_v1.CTRL_S1_SEED``), for rt_r0_s2
SECOND_SEED: Dict[str, int] = {"flat_seed": 1, "random_seed": 1, "random_seed_offset": 1000}
#: arm -> (phi key in :func:`ladder_phis`, or None = no reverse checkpoint; second seed?)
RT_PACKS: Dict[str, Dict[str, Tuple[Optional[str], bool]]] = {
    "r1": {"rt_r0": ("gold", False), "rt_r30": ("r30", False), "rt_r50": ("r50", False),
           "rt_r100": ("r100", False)},
    "r2": {"cold_ctl": (None, False), "rt_perm": ("permphi", False), "rt_r70": ("r70", False),
           "rt_r0_s2": ("gold", True)},
}
#: phase amendment A4's bands at ep8 (read by hand; no job applies them)
LIFT_PER = 0.50
PARTIAL_PER = 0.8164


def tag(rho: float) -> str:
    return f"r{int(round(rho * 100))}"


_SEED_KEYS = ("gold_json", "ids_json", "train_segments", "cv_segments")


def _seed(seed_inputs: Dict[str, Any]) -> Dict[str, Any]:
    missing = sorted(set(_SEED_KEYS) - set(seed_inputs))
    assert not missing, f"seed inputs: missing {missing} (keys {list(_SEED_KEYS)})"
    return {k: seed_inputs[k] for k in _SEED_KEYS}


# -- the phi fits ---------------------------------------------------------------------------------
def _gold_fit(labels_json: tk.Path, t: str, *, seed_inputs: Dict[str, Any], units_hdfs: Sequence[tk.Path],
              eta_npz: tk.Path, alias: Optional[str]) -> dict:
    """``PhoneTargetHdfJob`` -> the gold phi's fit on ``labels_json`` (the label source the only delta)."""
    from ..data.gold import PhoneTargetHdfJob
    from .supervised import blankfree_supervised_reverse_init
    from .supervised_steps import FIT

    seed = _seed(seed_inputs)
    # the gold targets' arguments (ids_json = the seed ids, label_key = None), labels swapped
    targets = PhoneTargetHdfJob(labels_json=labels_json, ids_json=seed["ids_json"], label_key=None)
    if alias:
        targets.add_alias(f"{alias}/{t}/targets")
    data, phi = blankfree_supervised_reverse_init(
        gold_json=labels_json, ids_json=seed["ids_json"], targets_hdf=targets.out_hdf,
        train_segments=seed["train_segments"], cv_segments=seed["cv_segments"],
        units_hdfs=units_hdfs, eta_npz=eta_npz, alias=f"{alias}/{t}/phi_init" if alias else None)
    return {"targets": targets, "phi_data": data, "phi": phi, "checkpoint": phi.out_checkpoints[FIT.epochs].path}


def corrupted_phi(rho: float, *, seed_inputs: Dict[str, Any], units_hdfs: Sequence[tk.Path], eta_npz: tk.Path,
                  alias: Optional[str] = ALIAS) -> dict:
    """ANALYSIS ONLY (uses transcripts).  :class:`CorruptSeedGoldJob` -> ``PhoneTargetHdfJob`` -> the
    gold phi's fit on the corrupted strings.

    :param seed_inputs: ``gold_json`` (``seed_gold_phones.json``), ``ids_json`` (``seed_ids.json``),
        ``train_segments`` / ``cv_segments`` (the seed split).
    :param units_hdfs: / :param eta_npz: the bed's rVAD-masked train unit HDFs and its eta table (the
        gold phi's own).
    :return: ``{"corrupt", "targets", "phi_data", "phi", "checkpoint"}``; ``checkpoint`` is epoch 8.
    """
    seed = _seed(seed_inputs)
    t = tag(rho)
    corr = CorruptSeedGoldJob(gold_json=seed["gold_json"], train_segments=seed["train_segments"],
                              cv_segments=seed["cv_segments"], rho=rho, seed=CORRUPTION_SEED)
    if alias:
        corr.add_alias(f"{alias}/{t}/corrupt")
    return {"corrupt": corr, **_gold_fit(corr.out_phones, t, seed_inputs=seed, units_hdfs=units_hdfs,
                                         eta_npz=eta_npz, alias=alias)}


def permuted_phi(*, seed_inputs: Dict[str, Any], units_hdfs: Sequence[tk.Path], eta_npz: tk.Path,
                 alias: Optional[str] = ALIAS) -> dict:
    """ANALYSIS ONLY (uses transcripts).  permphi: :class:`PermuteSeedGoldJob` (seed :data:`PERM_SEED`)
    -> targets -> the gold phi's fit.  Arguments and return as :func:`corrupted_phi`'s (``"permute"``)."""
    seed = _seed(seed_inputs)
    t = "permphi"
    perm = PermuteSeedGoldJob(gold_json=seed["gold_json"], seed=PERM_SEED)
    if alias:
        perm.add_alias(f"{alias}/{t}/permute")
    return {"permute": perm, **_gold_fit(perm.out_phones, t, seed_inputs=seed, units_hdfs=units_hdfs,
                                         eta_npz=eta_npz, alias=alias)}


def build_fits(*, seed_inputs: Dict[str, Any], units_hdfs: Sequence[tk.Path], eta_npz: tk.Path,
               phi_c_source=None, alias: Optional[str] = ALIAS) -> dict:
    """ANALYSIS ONLY (uses transcripts).  The fit stage: corruption, permutation, targets, the phi fits
    (``"<tag>/<key>"``, ``"permphi/<key>"``) and, when ``phi_c_source`` is given, phi_c's reverse block
    (``"phi_c"``, ``phi_first.phi_c_phi``)."""
    out: Dict[str, Any] = {}
    for rho in RHOS:
        for key, job in corrupted_phi(rho, seed_inputs=seed_inputs, units_hdfs=units_hdfs, eta_npz=eta_npz,
                                      alias=alias).items():
            out[f"{tag(rho)}/{key}"] = job
    for key, job in permuted_phi(seed_inputs=seed_inputs, units_hdfs=units_hdfs, eta_npz=eta_npz,
                                 alias=alias).items():
        out[f"permphi/{key}"] = job
    if phi_c_source is not None:
        from .phi_first import phi_c_phi

        out["phi_c"] = phi_c_phi(phi_c_source, alias=alias)
    return out


def ladder_phis(fits: dict, *, gold_phi) -> Dict[str, Optional[tk.Path]]:
    """The competence phi set: gold, r30 .. r100, phi_c (when fitted), permphi, and the random-init phi
    (None: no standalone checkpoint of it exists and no writer for one is registered).

    :param fits: :func:`build_fits`' output.
    :param gold_phi: the gold phi's checkpoint (``supervised.blankfree_supervised_reverse_init`` on the
        seed gold, epoch 8).
    """
    phis: Dict[str, Optional[tk.Path]] = {"gold": gold_phi}
    phis.update({tag(rho): fits[f"{tag(rho)}/checkpoint"] for rho in RHOS})
    if "phi_c" in fits:
        phis["phi_c"] = fits["phi_c"]
    phis["permphi"] = fits["permphi/checkpoint"]
    phis["random_init"] = None  # UNDETERMINED: no standalone checkpoint
    return phis


def competence_reads(phis: Dict[str, Optional[tk.Path]], *, reads: Dict[str, Any],
                     alias: Optional[str] = ALIAS) -> dict:
    """Statistic (a) and the posterior decode per phi on the CV holdout (``genmarg.genmarg_reads``;
    statistic (b) and dev-other are not ported).  Read name = the phi key.

    :param reads: ``phi_first.reads_bed(data)`` (the bed and the CV-holdout segment list).
    :return: ``{phi key: genmarg_reads(...)}`` plus ``"_not_read"`` (key -> reason).
    """
    from .genmarg import genmarg_reads

    out: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    for key, phi in phis.items():
        if phi is None:
            skipped[key] = ("genmarg_get_model loads phi only from a checkpoint; no random-init "
                            "checkpoint exists and the builder builds none from a seed")
            continue
        out[key] = genmarg_reads(phi, key, alias=f"{alias}/competence/{key}" if alias else None, **reads)
    out["_not_read"] = skipped
    return out


# -- the theta-at-cold-init arms (nodes R1 / R2) --------------------------------------------------
def schedules() -> Tuple[List[float], List[float]]:
    """``(temperature_schedule, learning_rates)`` of length 8: tau HELD at the phase's flat end value
    2.0, lr the phase's N = 20 list read at sub-epochs 1-8 (``diag._schedules``).

    The phase's 8.0 -> 2.0 anneal is still read and asserted but not run: at tau 8.0 the k2
    intersection overflows int32.
    """
    from ..training.config import THETA_LEARNING_RATE
    from ..training.schedules import phase_schedules

    tau_full, lr_full = phase_schedules(20)
    tau_annealed = [float(v) for v in tau_full[:NUM_SUBEPOCHS]]
    lr = [float(v) for v in lr_full[:NUM_SUBEPOCHS]]
    assert len(tau_annealed) == len(lr) == NUM_SUBEPOCHS, (len(tau_annealed), len(lr))
    assert tau_annealed[0] == 8.0 and (
        tau_annealed[0] > tau_annealed[1] > tau_annealed[2] > tau_annealed[3]), tau_annealed
    assert tau_annealed[3:] == [2.0] * (NUM_SUBEPOCHS - 3), tau_annealed
    tau = [float(tau_annealed[-1])] * NUM_SUBEPOCHS
    peak = float(THETA_LEARNING_RATE)
    assert lr[1:] == [peak] * (NUM_SUBEPOCHS - 1), lr
    assert 0.0 < lr[0] < peak, lr
    return tau, lr


def rt_model_args_delta(*, graph: Dict[str, Any]) -> Dict[str, Any]:
    """The k2 block of D10e's ``sup_k2lat`` plus ``lexlat_k2_chunk_seqs`` = :data:`RT_K2_CHUNK_SEQS`.

    :param graph: the in-house word graph (``{"hlg", "stats", "resources", "expected_build"}``,
        ``expected_build["theta"]`` 0.0; banked ``LexlatHLGBuildJob.cdcxYJMjiYj5``).
    """
    from ..training.arms import arm_graph
    from ..training.config import lexlat_k2_model_args

    g = arm_graph(graph, "inhouse")
    return lexlat_k2_model_args(hlg=g["hlg"], stats=g["stats"], resources=g["resources"],
                                expected_build=g["expected_build"], max_active=K2_MAX_ACTIVE,
                                onset=K2_ONSET, chunk_seqs=RT_K2_CHUNK_SEQS)


def rt_train_config(*, data: Dict[str, Any], graph: Dict[str, Any], phi_checkpoint: Optional[tk.Path],
                    second_seed: bool):
    """ANALYSIS ONLY when ``phi_checkpoint`` is a label-fitted phi.  D10e's ``supphi_k2lat`` with theta
    at the cold init (module doc).

    Deltas against the bed at :func:`schedules`: ``flat_checkpoint`` = ``FlatRecognizerInitJob(
    net_args=NET_ARGS, seed=0)`` (``seed=SECOND_SEED["flat_seed"]`` with ``second_seed``),
    ``reverse_checkpoint_path`` = ``phi_checkpoint`` (none when None), the k2 block
    (:func:`rt_model_args_delta`), and with ``second_seed`` :data:`SECOND_SEED`'s ``random_seed`` /
    ``random_seed_offset``.

    :param data: the bed's data dict (``training.arms``' keys); its ``flat_checkpoint`` is replaced.
    """
    from ..training.arms import arm_data
    from ..training.config import NET_ARGS, build_train_config
    from ..training.init import FlatRecognizerInitJob

    kw = arm_data(data)
    tau, lr = schedules()
    flat_seed = SECOND_SEED["flat_seed"] if second_seed else 0
    kw["flat_checkpoint"] = FlatRecognizerInitJob(net_args=NET_ARGS, seed=flat_seed).out_checkpoint
    seeds: Dict[str, Any] = {}
    if second_seed:
        assert set(SECOND_SEED) == {"flat_seed", "random_seed", "random_seed_offset"}, SECOND_SEED
        seeds = {"random_seed": SECOND_SEED["random_seed"],
                 "random_seed_offset": SECOND_SEED["random_seed_offset"]}
    return build_train_config(**kw, num_subepochs=NUM_SUBEPOCHS, temperature_schedule=tau, learning_rates=lr,
                              reverse_checkpoint_path=phi_checkpoint, lexlat_k2=rt_model_args_delta(graph=graph),
                              **seeds)


def build_rt(*, data: Dict[str, Any], graph: Dict[str, Any], phis: Dict[str, Optional[tk.Path]],
             alias: Optional[str] = ALIAS) -> Dict[str, Any]:
    """Nodes R1 / R2 (module doc): one ``train_arm`` per arm of :data:`RT_PACKS`, theta at the cold init.

    :param phis: :func:`ladder_phis`' set (keys ``gold``, ``r30`` .. ``r100``, ``permphi``).
    :return: ``{arm: ReturnnTrainingJob}``.
    """
    from ..training.jobs import TIME_RQMT, train_arm

    out: Dict[str, Any] = {}
    for node, arms in RT_PACKS.items():
        for arm, (phi_key, second_seed) in arms.items():
            phi = None if phi_key is None else phis[phi_key]
            assert phi_key is None or phi is not None, (arm, phi_key)
            cfg = rt_train_config(data=data, graph=graph, phi_checkpoint=phi, second_seed=second_seed)
            out[arm] = train_arm(f"{node}/{arm}/training", cfg, NUM_SUBEPOCHS, keep_epochs=KEEP_EPOCHS,
                                 time_rqmt=TIME_RQMT, alias_prefix=alias)
    return out
