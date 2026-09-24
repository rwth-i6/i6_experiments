"""T3.7 (test plan 2026-09-24): opt-in checks of the REAL input artefacts of phase 4A.

Run after the P0 input graph has produced its outputs::

    SAE_ARTEFACT_DIR=<setup>/output pytest -m artefact tests/test_artefacts.py

``SAE_ARTEFACT_DIR`` is the directory holding the registered output names (``sae/4a/lm/prior.npz``,
``sae/4a/data/vad/manifest.json``, ``sae/4a/data/speaker_eta/eta.stats.txt``).  Unset: every
``artefact`` test is skipped by ``conftest.py``.  Set, but an output not produced yet: that test
skips and names the missing file.  Files that are not registered (``prior.json``, ``eta.npz``, the
VAD ``raw_index`` HDFs, the prior window) are reached from a registered output through its job's
output directory (the registered name is a symlink into it) or through the path the artefact
itself records (the prior's ``meta["corpus"]``).

rho (S8).  The config's ``rate_rho_hz = 9.6619373279`` is a literal.  Its banked provenance
(``exp_logs/SAE/reports/review_rate_term_2026-09-15.md`` V7) is the FULL boundary-free
phonemisation T_phi: 2,784,159,269 phones / 778,025,128 words x 2.7 words/s.  It was NOT computed on
the prior's 1,010,000-line window, and the window (SIL-augmented, SIL at only about half of the word
boundaries) does not carry its word count, so the window cannot reproduce the literal to 1e-9.
What is checked instead:

* (no artefact) the literal against the banked counts, 1e-9 relative, and every built config
  carries it;
* (artefact) the numerator: the full SIL-augmented phone corpus the window was drawn from has
  exactly 2,784,159,269 non-SIL tokens on 39,630,169 lines (its job's ``stats.txt``);
* (artefact, slow) the window estimate: non-SIL phones / (lines + 2 x internal SILs), an unbiased
  word count under the p = 0.5 insertion, times 2.7, within 4 standard errors (the ratio's delta-
  method error from the window's own lines; T3.5's 4-sigma convention) of the literal.
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.data import speaker as SP
from i6_experiments.users.wu.experiments.unsupervised_asr.data.vad import BANKED_VAD_COUNTS
from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_prior as PP

RHO_HZ = 9.6619373279  # training/config.py model_args["rate_rho_hz"]
WORDS_PER_SEC = 2.7  # the disclosed read-speech constant (model/rate_term.py docstring)
# review_rate_term_2026-09-15.md V7 (analysis/out/rho.rate_term.txt of the source)
T_PHI_PHONES = 2_784_159_269
T_PHI_WORDS = 778_025_128
# model/prior.py docstring: PhonemizeWithSilJob.DbFgvZOGZQ8F, 40,418,261 -> 39,630,169 lines
PHONE_CORPUS_LINES_IN = 40_418_261
PHONE_CORPUS_LINES_OUT = 39_630_169
# PhoneNgramPriorJob.RtzbESkOedsT prior.stats.txt held_ppl_order3 (SAE_i6_ref.md: "held-out ppl 9.56";
# exp_logs/SAE_i6 prior entry: 9.561), compared at the 3 decimals it is banked with
BANKED_HELD_PPL3 = 9.561


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _artefact(rel: str) -> str:
    root = os.environ.get("SAE_ARTEFACT_DIR")
    if not root:
        pytest.skip("artefact: SAE_ARTEFACT_DIR is not set")
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        pytest.skip(f"artefact {rel} not produced yet under {root}")
    return path


def _job_output_dir(registered: str) -> str:
    """The job output directory a registered output (a symlink) points into."""
    return os.path.dirname(os.path.realpath(registered))


def _read_stats_txt(path: str) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _prior_meta(prior_npz: str) -> dict:
    d = np.load(prior_npz, allow_pickle=False)
    return ast.literal_eval(str(d["meta"][0]))


# ------------------------------------------------------------------------------------------------
# rho: the literal (no artefact needed)
# ------------------------------------------------------------------------------------------------
def test_rho_literal_is_the_banked_t_phi_ratio():
    rho = WORDS_PER_SEC * T_PHI_PHONES / T_PHI_WORDS
    assert abs(rho - RHO_HZ) <= 1e-9 * RHO_HZ, rho


def test_rho_literal_is_what_the_config_carries():
    from i6_experiments.users.wu.experiments.unsupervised_asr.training import config as C
    from i6_experiments.users.wu.experiments.unsupervised_asr.training.jobs import get_model_args

    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    cfg = C.build_train_config(
        train_feature_hdfs=[p("f.hdf")], train_units_hdfs=[p("u.hdf")], train_original_hdfs=[p("o.hdf")],
        dev_feature_hdfs=[p("f.hdf")], dev_units_hdfs=[p("u.hdf")], dev_original_hdfs=[p("o.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat.pt"))
    assert get_model_args(cfg)["rate_rho_hz"] == RHO_HZ


# ------------------------------------------------------------------------------------------------
# the prior
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prior_npz():
    return _artefact("sae/4a/lm/prior.npz")


@pytest.fixture(scope="module")
def window_path(prior_npz):
    meta = _prior_meta(prior_npz)
    path = meta["corpus"]
    if not os.path.exists(path):
        pytest.skip(f"the prior's window {path} (prior meta 'corpus') is not readable here")
    return path


@pytest.mark.artefact
def test_prior_tables(prior_npz):
    prior = PP.PhoneNgramPrior.load(prior_npz)
    assert prior.log_uni.shape == (PP.N_TYPES,)
    assert prior.log_bi.shape == (PP.N_CTX, PP.N_TYPES)
    assert prior.log_tri.shape == (PP.N_CTX * PP.N_CTX, PP.N_TYPES)
    d = np.load(prior_npz, allow_pickle=False)
    assert [str(p) for p in d["phones"]] == list(PP.PHONE2ID)
    assert np.all(np.isfinite(prior.log_uni)) and np.all(np.isfinite(prior.log_bi))
    assert np.all(np.isfinite(prior.log_tri))
    np.testing.assert_allclose(np.exp(prior.log_uni).sum(), 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.exp(prior.log_bi).sum(1), 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.exp(prior.log_tri).sum(1), 1.0, rtol=0, atol=1e-9)
    # every line of the phone corpus opens with <SIL>
    p_sil_bos = float(np.exp(prior.log_tri[PP.BOS_ID * PP.N_CTX + PP.BOS_ID, PP.SIL_ID]))
    assert p_sil_bos > 0.9, p_sil_bos
    meta = _prior_meta(prior_npz)
    assert meta["lines_read"] == PP.DEFAULT_COUNT_LINES + PP.DEFAULT_HELD_LINES
    assert meta["held_stride"] == PP.HELD_STRIDE and meta["lines_held"] == PP.DEFAULT_HELD_LINES


@pytest.mark.artefact
def test_prior_json_held_perplexity(prior_npz):
    js = os.path.join(_job_output_dir(prior_npz), "prior.json")
    if not os.path.exists(js):
        pytest.skip(f"{js} not found next to the registered prior.npz")
    stats = json.load(open(js))
    assert round(stats["held_ppl_order3"], 3) == BANKED_HELD_PPL3, stats["held_ppl_order3"]


@pytest.mark.artefact
@pytest.mark.slow
def test_prior_held_perplexity_recomputed(prior_npz, window_path):
    """The npz on disk scores its own 10,000 held lines (every 101st line of the window) at the
    banked perplexity, and at the job's own number when ``prior.json`` is there."""
    prior = PP.PhoneNgramPrior.load(prior_npz)
    n_read = PP.DEFAULT_COUNT_LINES + PP.DEFAULT_HELD_LINES
    held = []
    for i, toks in enumerate(PP.read_phone_lines(window_path, limit=n_read)):
        if toks and i % PP.HELD_STRIDE == 0:
            held.append(PP._to_ids(toks))
    held = held[:PP.DEFAULT_HELD_LINES]
    assert len(held) == PP.DEFAULT_HELD_LINES
    ppl = prior.perplexity(held, 3)
    assert round(ppl, 3) == BANKED_HELD_PPL3, ppl
    js = os.path.join(_job_output_dir(prior_npz), "prior.json")
    if os.path.exists(js):
        assert abs(ppl - json.load(open(js))["held_ppl_order3"]) <= 1e-9 * ppl


# ------------------------------------------------------------------------------------------------
# rho from the artefacts
# ------------------------------------------------------------------------------------------------
@pytest.mark.artefact
def test_rho_numerator_on_the_full_phone_corpus(window_path):
    """The window's source corpus (named in the window job's stats.txt) has T_phi's exact non-SIL
    token count and line count."""
    head = open(os.path.join(os.path.dirname(window_path), "stats.txt")).readline()
    assert head.startswith("seeded uniform line sample <- "), head
    corpus = head.split("<- ", 1)[1].strip()
    stats_path = os.path.join(os.path.dirname(corpus), "stats.txt")
    if not os.path.exists(stats_path):
        pytest.skip(f"{stats_path} (the phone corpus job's stats) is not readable here")
    st = _read_stats_txt(stats_path)
    assert (float(st["sil_prob"]), st["surround"], int(st["seed"])) == (0.5, "True", 0)
    assert int(st["lines_in"]) == PHONE_CORPUS_LINES_IN
    assert int(st["lines_out"]) == PHONE_CORPUS_LINES_OUT
    assert int(st["tokens"]) - int(st["sil_tokens"]) == T_PHI_PHONES


@pytest.mark.artefact
@pytest.mark.slow
def test_rho_estimated_on_the_prior_window(window_path):
    n_read = PP.DEFAULT_COUNT_LINES + PP.DEFAULT_HELD_LINES
    phones, words = [], []
    for toks in PP.read_phone_lines(window_path, limit=n_read):
        n_sil = sum(1 for t in toks if t == "<SIL>")
        assert toks[0] == "<SIL>" and toks[-1] == "<SIL>" and len(toks) >= 3
        phones.append(len(toks) - n_sil)
        words.append(1 + 2 * (n_sil - 2))  # E[internal SIL] = (W - 1) / 2
    assert len(phones) == n_read
    p = np.asarray(phones, dtype=np.float64)
    w = np.asarray(words, dtype=np.float64)
    r = p.sum() / w.sum()
    z = p - r * w
    se = math.sqrt(len(z) * z.var(ddof=1)) / w.sum()
    rho = WORDS_PER_SEC * r
    assert abs(rho - RHO_HZ) <= 4 * WORDS_PER_SEC * se, (rho, RHO_HZ, WORDS_PER_SEC * se)


# ------------------------------------------------------------------------------------------------
# VAD manifest and eta
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vad_manifest():
    return _artefact("sae/4a/data/vad/manifest.json")


@pytest.mark.artefact
def test_vad_manifest_is_the_banked_counts(vad_manifest):
    summary = json.load(open(vad_manifest))["summary"]
    assert set(summary) == set(BANKED_VAD_COUNTS)
    for split, banked in BANKED_VAD_COUNTS.items():
        got = {k: summary[split][k] for k in banked}
        assert got == banked, (split, got, banked)
        assert summary[split]["short_or_empty"] == []


def _vad_tags(vad_manifest):
    import h5py

    out = {}
    d = _job_output_dir(vad_manifest)
    for split in BANKED_VAD_COUNTS:
        files = sorted(glob.glob(os.path.join(d, f"raw_index.{split}.shard*.hdf")))
        if not files:
            pytest.skip(f"no raw_index.{split}.shard*.hdf in {d}")
        tags = []
        for f in files:
            with h5py.File(f, "r") as fh:
                tags += [t.decode() if isinstance(t, bytes) else str(t) for t in fh["seqTags"][:]]
        out[split] = tags
    return out


@pytest.mark.artefact
def test_eta_covers_train_and_dev(vad_manifest):
    stats = _artefact("sae/4a/data/speaker_eta/eta.stats.txt")
    eta_npz = os.path.join(_job_output_dir(stats), "eta.npz")
    if not os.path.exists(eta_npz):
        pytest.skip(f"{eta_npz} not found next to the registered eta.stats.txt")
    d = np.load(eta_npz, allow_pickle=False)
    tags = [str(t) for t in d["tags"]]
    eta = d["eta"]
    assert eta.dtype == np.float32 and eta.shape == (len(tags), SP.ETA_DIM)
    assert np.all(np.isfinite(eta))
    assert tags == sorted(tags) and len(set(tags)) == len(tags)
    assert len(tags) == SP._EXPECT_FIT_UTTS + SP._EXPECT_DEV_UTTS
    vad_tags = _vad_tags(vad_manifest)
    for split, ts in vad_tags.items():
        assert len(ts) == BANKED_VAD_COUNTS[split]["utterances"], split
        missing = set(ts) - set(tags)
        assert not missing, (split, len(missing), sorted(missing)[:3])
