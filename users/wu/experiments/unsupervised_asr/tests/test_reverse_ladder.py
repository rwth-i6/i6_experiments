"""Tests of ``reverse_model/ladder.py`` (ANALYSIS ONLY, uses transcripts): the corruption and the
permutation rules on synthetic gold strings, the local ``edit_counts``, and the theta-at-cold-init
arm deltas at graph-build level (no job runs)."""

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.phones import ARPABET_39
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import ladder as L
from i6_experiments.users.wu.experiments.unsupervised_asr.training.init import FlatRecognizerInitJob
from i6_experiments.users.wu.experiments.unsupervised_asr.training.jobs import get_model_args


def _gold(n_utts=40, seed=3):
    rng = np.random.default_rng(seed)
    return {f"utt{i:03d}": [ARPABET_39[j] for j in rng.integers(0, 39, size=int(rng.integers(5, 30)))]
            for i in range(n_utts)}


def test_corruption_counts_length_and_symbols():
    gold = _gold()
    for rho in L.RHOS:
        corrupted, subs = L.corrupt_all(gold, rho, L.CORRUPTION_SEED)
        assert list(corrupted) == list(gold)
        for t, seq in gold.items():
            assert len(corrupted[t]) == len(seq)
            assert subs[t] == int(round(rho * len(seq)))
            # every substituted position differs from the original (the original is renormalised away)
            assert sum(a != b for a, b in zip(corrupted[t], seq)) == subs[t]
            assert set(corrupted[t]) <= set(ARPABET_39)


def test_corruption_is_nested_in_rho_and_order_free():
    gold = _gold()
    lo, _ = L.corrupt_all(gold, 0.3, 0)
    hi, _ = L.corrupt_all(gold, 0.7, 0)
    for t, seq in gold.items():
        changed_lo = {i for i, (a, b) in enumerate(zip(lo[t], seq)) if a != b}
        assert all(hi[t][i] == lo[t][i] for i in changed_lo)
    # the per-utterance rng depends on (seed, tag) only: reversing the key order changes nothing
    rev = dict(reversed(list(gold.items())))
    again, _ = L.corrupt_all(rev, 0.3, 0)
    assert all(again[t] == lo[t] for t in gold)
    other, _ = L.corrupt_all(gold, 0.3, 1)
    assert any(other[t] != lo[t] for t in gold)


def test_unknown_symbol_refused():
    with pytest.raises(AssertionError):
        L.gold_unigram({"u": ["AA", "SIL"]})


def test_corruption_stats_per_le_rate():
    gold = _gold()
    corrupted, subs = L.corrupt_all(gold, 0.5, 0)
    tags = list(gold)
    rec = L.corruption_stats(gold, corrupted, subs, {"all": tags, "train": tags[:30], "held": tags[30:]})
    assert rec["all"]["per"] <= rec["all"]["realised_substitution_rate"] + 1e-12
    assert rec["all"]["collapses"] == 0
    assert rec["train"]["substituted"] + rec["held"]["substituted"] == rec["all"]["substituted"]


def test_edit_counts():
    assert L.edit_counts(["A", "B", "C"], ["A", "B", "C"]) == (0, 0, 0)
    assert L.edit_counts(["A", "X", "C"], ["A", "B", "C"]) == (1, 0, 0)
    assert L.edit_counts(["A", "C"], ["A", "B", "C"]) == (0, 1, 0)
    assert L.edit_counts(["A", "B", "B", "C"], ["A", "B", "C"]) == (0, 0, 1)
    assert sum(L.edit_counts([], ["A", "B"])) == 2


def test_permutation_bijection_and_inverse():
    mapping = L.phone_permutation(L.PERM_SEED)
    assert sorted(mapping) == sorted(ARPABET_39) and sorted(mapping.values()) == sorted(ARPABET_39)
    assert mapping == L.phone_permutation(0)
    gold = _gold()
    gold["with_sil"] = ["AA", "SIL", "B"]
    permuted, unmapped = L.permute_all(gold, mapping)
    assert unmapped == 1 and permuted["with_sil"][1] == "SIL"
    inverse = {v: k for k, v in mapping.items()}
    assert all([inverse.get(p, p) for p in permuted[t]] == gold[t] for t in gold)


def test_job_arguments():
    p = tk.Path("/nonexistent/x")
    with pytest.raises(AssertionError):
        L.CorruptSeedGoldJob(gold_json=p, train_segments=p, cv_segments=p, rho=0.0)
    job = L.CorruptSeedGoldJob(gold_json=p, train_segments=p, cv_segments=p, rho=0.5)
    assert job.seed == 0 and job.rho == 0.5
    assert L.PermuteSeedGoldJob(gold_json=p).seed == 0
    assert [L.tag(r) for r in L.RHOS] == ["r30", "r50", "r70", "r100"]


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )


GRAPH = dict(hlg=tk.Path("/nonexistent/HLG.pt"), stats=tk.Path("/nonexistent/build.json"),
             resources=tk.Path("/nonexistent/lexlat_resources.npz"),
             expected_build={"backoff_loops": "word_boundary", "escape": True, "sil_prob": 0.5, "theta": 0.0})


def test_schedules():
    tau, lr = L.schedules()
    assert tau == [2.0] * 8
    assert lr == [1e-05] + [1e-4] * 7


def test_rt_arm_deltas():
    phi = tk.Path("/nonexistent/phi.pt")
    cfg = L.rt_train_config(data=_data(), graph=GRAPH, phi_checkpoint=phi, second_seed=False)
    args = get_model_args(cfg)
    assert args["reverse_checkpoint_path"] is phi
    assert args["lexlat_k2_chunk_seqs"] == 4 and args["lexlat_k2_onset"] == 1
    assert args["lexlat_k2_max_active"] == 1000 and args["lexlat_k2_ramp"] == 3
    flat = args["recognizer_checkpoint_path"]
    assert isinstance(flat.creator, FlatRecognizerInitJob) and flat.creator.seed == 0
    assert "random_seed" not in cfg.config
    assert "random_seed_offset" not in cfg.config["train"]["dataset"]["datasets"]["feats"]

    cold = get_model_args(L.rt_train_config(data=_data(), graph=GRAPH, phi_checkpoint=None, second_seed=False))
    assert "reverse_checkpoint_path" not in cold

    s2 = L.rt_train_config(data=_data(), graph=GRAPH, phi_checkpoint=phi, second_seed=True)
    assert s2.config["random_seed"] == 1
    assert s2.config["train"]["dataset"]["datasets"]["feats"]["random_seed_offset"] == 1000
    assert get_model_args(s2)["recognizer_checkpoint_path"].creator.seed == 1

    with pytest.raises(AssertionError):  # the official graph (theta 5.0) is not the ladder's
        L.rt_train_config(data=_data(), graph={**GRAPH, "expected_build": {**GRAPH["expected_build"], "theta": 5.0}},
                          phi_checkpoint=phi, second_seed=False)


def test_build_rt_and_fits():
    phi = tk.Path("/nonexistent/phi.pt")
    phis = {"gold": phi, "r30": phi, "r50": phi, "r70": phi, "r100": phi, "permphi": phi}
    jobs = L.build_rt(data=_data(), graph=GRAPH, phis=phis, alias=None)
    assert sorted(jobs) == sorted(a for arms in L.RT_PACKS.values() for a in arms)
    assert all(j.rqmt["time"] == 11.5 and j.rqmt["gpu"] == 1 for j in jobs.values())
    seed = {k: tk.Path(f"/nonexistent/{k}") for k in ("gold_json", "ids_json", "train_segments", "cv_segments")}
    fits = L.build_fits(seed_inputs=seed, units_hdfs=[tk.Path("/nonexistent/u.hdf")],
                        eta_npz=tk.Path("/nonexistent/eta.npz"), alias=None)
    assert fits["r50/corrupt"].rho == 0.5 and fits["permphi/permute"].seed == 0
    assert fits["r30/targets"].labels_json.get_path() == fits["r30/corrupt"].out_phones.get_path()
    assert fits["r30/targets"].label_key is None
    phis = L.ladder_phis(fits, gold_phi=phi)
    assert phis["random_init"] is None and phis["r100"] is fits["r100/checkpoint"]
