"""Tests of the LABEL-USING, REPORT-ONLY D4 dev-other phone read of ``reverse_model/genmarg.py`` (G0.G,
``SAE_i6_P0.md``): the D4 sample, the alignment, the Hungarian map, PER, NMI and the frame labels
against hand-computed oracles, E[d] of a uniform table, the uniform prior npz, the graph of
:func:`dev_other_reads` and its quarantine.  CPU only; no job runs on a cluster."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.analysis.per import edit_counts
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import ARPABET_39, PHONES, SIL
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg as G
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import phi_first as pf


def _redirect(job, tmp: Path):
    """Point every ``out_*`` path of ``job`` into ``tmp`` so ``run()`` can execute locally."""
    for name, val in list(vars(job).items()):
        if name.startswith("out_") and isinstance(val, tk.Path):
            setattr(job, name, tk.Path(str(tmp / Path(val.get_path()).name)))
    return job


# -- the D4 sample ------------------------------------------------------------------------------
def test_d4_sample_nonempty_gold_seed0(tmp_path):
    gold = {"dev-other": {"d": ["AA"], "a": ["B"], "e": [], "c": ["K", "S"], "b": ["T"], "f": []}}
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold))
    job = _redirect(G.GenMargSampleJob(gold=tk.Path(str(path)), split="dev-other", n=3, seed=0), tmp_path)
    job.run()
    # hand oracle: eligible (non-empty gold) sorted = [a, b, c, d]; RandomState(0).permutation(4) =
    # [2, 3, 1, 0]; the first 3 -> c, d, b
    assert (tmp_path / "sample.segments").read_text().split() == ["c", "d", "b"]
    rec = json.loads((tmp_path / "sample.json").read_text())
    assert rec["eligible"] == 4 and rec["realised"] == 3 and rec["split"] == "dev-other"


def test_d4_builder_uses_the_d4_rule():
    out = G.dev_other_reads(tk.Path("/nonexistent/phi.pt"), "x", **_devother_kwargs())
    s = out["sample"]
    assert (s.split, s.n, s.seed, s.segments) == ("dev-other", 500, 0, None)
    assert G.SELECT_N == 500


# -- alignment, PER, Hungarian ------------------------------------------------------------------
def test_align_matches_edit_counts():
    rng = np.random.default_rng(0)
    alphabet = ["AA", "B", "K", "SIL"]
    for _ in range(300):
        hyp = [alphabet[i] for i in rng.integers(0, 4, size=int(rng.integers(0, 9)))]
        ref = [alphabet[i] for i in rng.integers(0, 3, size=int(rng.integers(0, 9)))]
        pairs = G.align(hyp, ref)
        assert [h for h, _ in pairs if h is not None] == hyp
        assert [r for _, r in pairs if r is not None] == ref
        s = sum(h is not None and r is not None and h != r for h, r in pairs)
        d = sum(h is None for h, _ in pairs)
        i = sum(r is None for _, r in pairs)
        assert (s, d, i) == edit_counts(hyp, ref)


def test_corpus_per_hand():
    # utt1: hyp [A, C] vs ref [A, B, C] -> 1 deletion; utt2: hyp [X] vs ref [Y] -> 1 substitution
    out = G.corpus_per({"u1": ["A", "C"], "u2": ["X"]}, {"u1": ["A", "B", "C"], "u2": ["Y"]})
    assert (out["sub"], out["del"], out["ins"], out["ref_tokens"]) == (1, 1, 0, 4)
    assert out["per"] == pytest.approx(2 / 4)


def test_hungarian_hand_fixtures():
    # 2 phones + DELETE, 3 symbols (S plays SIL): gains a->y 3, b->x 2, S->x 1
    pairs = [("a", "y")] * 3 + [("b", "x")] * 2 + [("S", "x")] + [("a", None), (None, "x")]
    assert G.hungarian_map(pairs, ["a", "b", "S"], ["x", "y"]) == {"a": "y", "b": "x", "S": G.DELETE}
    # the square map has one DELETE slot and may spend it on a non-SIL symbol: a->x 5, S->y 3, b->y 1
    pairs = [("a", "x")] * 5 + [("S", "y")] * 3 + [("b", "y")]
    assert G.hungarian_map(pairs, ["a", "b", "S"], ["x", "y"]) == {"a": "x", "S": "y", "b": G.DELETE}
    assert G.relabel(["a", "S", "b", "a"], {"a": "x", "S": "y", "b": G.DELETE}) == ["x", "y", "x"]
    with pytest.raises(AssertionError):
        G.hungarian_map(pairs, ["a", "b"], ["x", "y"])


def _strings(n=30, seed=1):
    rng = np.random.default_rng(seed)
    return {f"u{i:02d}": [ARPABET_39[j] for j in rng.integers(0, 39, size=int(rng.integers(8, 25)))]
            for i in range(n)}


def test_score_gold_like_decode_is_identity():
    gold = _strings()
    # the decode is the gold string with SIL at the edges and between some tokens
    dec = {t: [SIL] + [x for k, p in enumerate(g) for x in ((p, SIL) if k % 4 == 0 else (p,))] + [SIL]
           for t, g in gold.items()}
    out = G.DevOtherPhoneReadJob.score(dec, gold)
    assert out["direct"]["per"] == 0.0 and out["hungarian"]["per"] == 0.0
    assert out["map_is_identity"] and out["symbols_mapped_to_identity"] == 40
    assert out["token_nmi"]["nmi"] == pytest.approx(1.0)


def test_score_recovers_a_permutation():
    gold = _strings()
    rng = np.random.default_rng(5)
    perm = dict(zip(ARPABET_39, [ARPABET_39[i] for i in rng.permutation(39)]))
    inv = {v: k for k, v in perm.items()}
    # no SIL: the identity-label alignment of a string with NO matches is positional only when the
    # lengths agree (edge SILs make shifted alignments tie, the A15-E weakness of the decode map)
    dec = {t: [inv[p] for p in g] for t, g in gold.items()}
    dec["dead"] = None
    gold["dead"] = ["AA", "B"]
    out = G.DevOtherPhoneReadJob.score(dec, gold)
    assert out["hungarian"]["per"] == 0.0 and out["direct"]["per"] > 0.5
    assert out["impossible_tags"] == ["dead"] and out["n_decoded"] == len(gold) - 1
    assert all(out["hungarian_map"][inv[p]] == p for p in ARPABET_39)
    assert out["hungarian_map"][SIL] == G.DELETE
    n = out["direct"]["ref_tokens"]
    errs = out["direct"]["sub"] + out["direct"]["del"] + out["direct"]["ins"]
    assert out["per_impossible_as_deletions"] == pytest.approx((errs + 2) / (n + 2))


def test_score_hand_substitution_and_repeats():
    gold = {"u": ["AA", "B", "K"]}
    dec = {"u": ["AA", SIL, "AA", "K"]}  # direct hyp [AA, AA, K]: 1 substitution, 1 adjacent repeat
    out = G.DevOtherPhoneReadJob.score(dec, gold)
    assert out["direct"]["per"] == pytest.approx(1 / 3)
    assert out["adjacent_repeats"] == 1 and out["direct_per_collapsed"] == pytest.approx(1 / 3)


# -- NMI ----------------------------------------------------------------------------------------
def test_nmi_oracles():
    perfect = [("a", "x"), ("b", "y"), ("c", "z")] * 4
    assert G.nmi_bits(perfect)["nmi"] == pytest.approx(1.0)
    indep = [(x, y) for x in "ab" for y in "xyz"] * 3
    assert G.nmi_bits(indep)["nmi"] == pytest.approx(0.0, abs=1e-12)
    # hand 2 x 2: counts (a,x)=2, (a,y)=1, (b,y)=1; n = 4
    hand = [("a", "x"), ("a", "x"), ("a", "y"), ("b", "y")]
    out = G.nmi_bits(hand)
    h_x = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    h_y = 1.0
    mi = 0.5 * math.log2(4 / 3) + 0.25 * math.log2(2 / 3) + 0.25 * 1.0
    assert out["h_x_bits"] == pytest.approx(h_x) and out["h_y_bits"] == pytest.approx(h_y)
    assert out["mi_bits"] == pytest.approx(mi)
    assert out["nmi"] == pytest.approx(mi / math.sqrt(h_x * h_y))
    assert G.nmi_bits([("a", "x")] * 3)["nmi"] is None
    assert G.nmi_bits([])["n"] == 0


# -- frame labels -------------------------------------------------------------------------------
def test_frame_labels_hand():
    # segments over 5 retained frames: AA x2, SIL x3
    assert G.segment_frame_labels([[0, 0, 2], [39, 2, 3]], 5) == ["AA", "AA", SIL, SIL, SIL]
    with pytest.raises(AssertionError):
        G.segment_frame_labels([[0, 0, 2]], 3)
    # retained frames at raw 50 Hz positions 0, 1, 2, 5: centres 0.01, 0.03, 0.05, 0.11 s
    iv = [("B", 0.04, 0.10), ("AA", 0.0, 0.04)]
    assert G.gold_frame_labels([0, 1, 2, 5], iv) == ["AA", "AA", "B", SIL]


# -- E[d] and the uniform prior -----------------------------------------------------------------
def test_duration_means_uniform_table():
    import torch

    out = G.duration_means(torch.zeros(40, 50))
    # zero logits: uniform on [2, 25] for phones (mean 13.5) and on [2, 50] for SIL (mean 26)
    assert out["phone_types_mean"] == pytest.approx(13.5) and out["sil"] == pytest.approx(26.0)


def test_uniform_prior_job(tmp_path):
    from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior

    job = _redirect(G.UniformPhonePriorJob(), tmp_path)
    job.run()
    prior = PhoneNgramPrior.load(str(tmp_path / "prior.npz"))
    for table in (prior.log_uni[None], prior.log_bi, prior.log_tri):
        assert np.allclose(table, -math.log(40)) and np.allclose(np.exp(table).sum(-1), 1.0)


# -- the graph and the quarantine ---------------------------------------------------------------
def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )


def _devother_kwargs(**kw):
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    stream = {"features": [p("do.feats.hdf")], "units": [p("do.units.hdf")], "originals": [p("do.orig.hdf")]}
    return dict(bed=pf.reads_bed(_data())["bed"], gold=p("gold.json"), stream=stream,
                raw_index_hdfs=[p("do.idx.hdf")], mfa_dir=p("mfa"), **kw)


def test_dev_other_reads_graph():
    kw = _devother_kwargs()
    bed = kw["bed"]
    out = G.dev_other_reads(tk.Path("/nonexistent/phi.pt"), "x", **kw)
    assert set(out) == {"sample", "uniform_prior", "trigram", "uniform"}
    tri, uni = out["trigram"]["decode"], out["uniform"]["decode"]
    assert tri.job_id() != uni.job_id()
    for dec in (tri, uni):
        data = dec.returnn_config.config["forward_data"]["datasets"]
        assert data["feats"]["files"] == kw["stream"]["features"]
        assert data["units"]["files"] == kw["stream"]["units"]
        assert data["original"]["files"] == kw["stream"]["originals"]
        assert data["feats"]["seq_list_filter_file"] == out["sample"].out_segments
        assert "seq_list_filter_file" not in data["units"] and "seq_list_filter_file" not in data["original"]
        assert dec.rqmt == {"gpu": 1, "cpu": 4, "mem": 32, "time": 2}
        cb = dec.returnn_config.python_epilog[0].serializer_objects[3]
        assert cb.hashed_arguments["dataset"] == "dev-other"
        assert cb.hashed_arguments["expected_utterances"] == 500
    args = lambda d: d.returnn_config.python_epilog[0].serializer_objects[1].hashed_arguments["model_args"]  # noqa: E731
    assert args(tri) == bed["model_args"]
    diff = {k for k in bed["model_args"] if args(uni)[k] != bed["model_args"][k]}
    assert diff == {"prior_npz_path"} and args(uni)["prior_npz_path"] == out["uniform_prior"].out_prior
    assert out["trigram"]["report"].gendecode == tri.out_files["gendecode.json"]
    # the bed's dev set is unchanged by the build (deep copy)
    assert bed["dev"]["datasets"]["feats"]["files"] == [tk.Path("/nonexistent/feats.0.hdf")]
    only = G.dev_other_reads(tk.Path("/nonexistent/phi.pt"), "x", **_devother_kwargs(priors=("trigram",)))
    assert set(only) == {"sample", "trigram"}
    assert only["trigram"]["decode"].job_id() == tri.job_id()
    with pytest.raises(ValueError):
        G.dev_other_reads(tk.Path("/nonexistent/phi.pt"), "x", **_devother_kwargs(priors=("bigram",)))


def test_quarantine():
    reads = pf.reads_bed(_data())
    with pytest.raises(ValueError):
        G.genmarg_reads(tk.Path("/nonexistent/phi.pt"), "x", datasets=("dev-other",), alias=None, **reads)
    # a dev-other decode record cannot enter the L2-1 selection
    tags = ["a", "b", "c"]
    rec = lambda v, shuffle=None: {  # noqa: E731
        "dataset": "cv_holdout", "settings": {"shuffle_seed": shuffle},
        "summary": {"expected_nonsil_rate": 1.0, "n_impossible": 0},
        "per_utterance": {t: {"nll_tau1_per_frame": v, "impossible": False, "expected_nonsil_tokens": 10.0,
                              "original_frames": 50} for t in tags}}
    dec = {"dataset": "dev-other", "settings": {"shuffle_seed": None},
           "per_utterance": {t: {} for t in tags}, "emitted_nonsil_rate": {"pooled_hz": 10.0, "n": 3}}
    restarts = {pf.arm_name(s): rec(5.0 + 0.01 * s) for s in range(1, 17)}
    kw = dict(restarts=restarts,
              nulls={pf.arm_name(s, "null"): rec(6.0, shuffle=0) for s in range(1, 5)},
              nulls_real={pf.arm_name(s, "null"): rec(7.0) for s in range(1, 5)},
              phi_c={pf.arm_name(s, "phic"): rec(5.5) for s in (1, 2)},
              reruns={"em_s01": rec(5.01), "em_s02": rec(5.02)},
              restart_decodes={k: dict(dec) for k in restarts})
    with pytest.raises(AssertionError):
        G.GenMargSelectionJob.decide(**kw)


def test_eval_dataset_guards():
    bed = pf.reads_bed(_data())["bed"]
    seg = tk.Path("/nonexistent/s.segments")
    with pytest.raises(AssertionError):
        G.eval_dataset(bed["dev"], "dev-other", seg)  # dev-other needs its stream
    with pytest.raises(AssertionError):
        G.eval_dataset(bed["dev"], "cv_holdout", seg, stream={"features": [seg]})
    with pytest.raises(ValueError):
        G.eval_dataset(bed["dev"], "dev-clean", seg)


def test_phone_read_job_runs_on_a_fixture(tmp_path):
    """``DevOtherPhoneReadJob.run`` end to end on a two-utterance fixture: a gendecode json, a
    raw-index HDF in the VAD job's layout, an MFA parquet and a reverse-only checkpoint."""
    import h5py
    import pandas as pd
    import torch

    tags = ["u1", "u2"]
    (tmp_path / "sample.segments").write_text("\n".join(tags) + "\n")
    gold = {"dev-other": {"u1": ["AA", "B"], "u2": ["K"]}}
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    # u1: SIL x2, AA x2, B x2 over 6 retained frames; u2 impossible
    dec = {"dataset": "dev-other", "name": "fx", "per_utterance": {
        "u1": {"impossible": False, "tokens": [SIL, "AA", "B"], "segments": [[39, 0, 2], [0, 2, 2], [2, 4, 2]]},
        "u2": {"impossible": True, "tokens": None, "segments": None}}}
    (tmp_path / "gendecode.json").write_text(json.dumps(dec))
    idx = {"u1": np.array([0, 1, 2, 3, 4, 5]), "u2": np.array([0, 1])}
    with h5py.File(tmp_path / "idx.hdf", "w") as fh:
        fh["seqTags"] = np.array([t.encode() for t in tags])
        fh["seqLengths"] = np.array([[len(idx[t])] for t in tags])
        fh["inputs"] = np.concatenate([idx[t] for t in tags])
    (tmp_path / "mfa" / "data").mkdir(parents=True)
    # centres 0.01 .. 0.11 s: SIL (uncovered) 0.01, 0.03; AA1 [0.04, 0.08); B [0.08, 0.2)
    rows = [{"id": "u1", "phonemes": [{"phoneme": "AA1", "start": 0.04, "end": 0.08},
                                      {"phoneme": "B", "start": 0.08, "end": 0.2}]},
            {"id": "u2", "phonemes": [{"phoneme": "K", "start": 0.0, "end": 0.1}]}]
    pd.DataFrame(rows).to_parquet(tmp_path / "mfa" / "data" / "dev_other-00000.parquet")
    torch.save({"model": {"dur_logits": torch.zeros(40, 50)}, "epoch": 8, "step": 1}, tmp_path / "phi.pt")

    p = lambda name: tk.Path(str(tmp_path / name))  # noqa: E731
    job = _redirect(G.DevOtherPhoneReadJob(
        gendecode=p("gendecode.json"), gold=p("gold.json"), sample_segments=p("sample.segments"),
        phi=p("phi.pt"), prior="trigram", raw_index_hdfs=[p("idx.hdf")], mfa_dir=p("mfa")), tmp_path)
    job.run()
    out = json.loads((tmp_path / "phone_read.json").read_text())
    assert out["label_using"] and out["report_only"] and out["prior"] == "trigram"
    assert out["direct"]["per"] == 0.0 and out["impossible_tags"] == ["u2"]
    assert out["per_impossible_as_deletions"] == pytest.approx(1 / 3)
    assert out["frame_nmi"]["n"] == 6 and out["frame_nmi"]["nmi"] == pytest.approx(1.0)
    assert out["e_d_table"]["phone_types_mean"] == pytest.approx(13.5)
    assert out["e_d_decode"]["nonsil"] == 2.0 and out["e_d_decode"]["sil"] == 2.0
    assert "Hungarian PER" in (tmp_path / "phone_read.txt").read_text()
