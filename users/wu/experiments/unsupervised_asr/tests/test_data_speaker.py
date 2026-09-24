"""T3.3 / T3.4 (test plan 2026-09-24): the speaker vector eta.

T3.3 -- ``fit_speaker_pca`` against an SVD oracle (orthonormal rows, the top right singular
vectors up to sign, the sign rule, unit-variance scaling, save/load round trip), and
``SpeakerEtaJob`` on tiny per-utterance stat pickles (sorted tags, rows = ``apply()``, the
fit / dev disjointness assertion).

T3.4 -- the RETURNN-side lookup: ``lookup_eta`` returns rows in batch-tag order and raises
``KeyError`` on an unknown tag, and inside ``train_step`` the segment table of a batch row follows
its TAG: permuting the batch permutes the tables, and each row equals
``build_segment_table`` of that row's own units and eta.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.data import speaker as SP

D, N, DIM = 12, 200, 3


def _pca_data(seed=0, n=N, d=D):
    """Gaussian cloud with well separated principal variances (so the eigenvectors are unique)."""
    rng = np.random.RandomState(seed)
    scales = np.array([9.0, 6.0, 4.0, 2.5] + [1.0] * (d - 4))[:d]
    q, _ = np.linalg.qr(rng.randn(d, d))
    return rng.randn(n, d) * scales @ q.T + rng.randn(d) * 3.0


# ------------------------------------------------------------------------------------------------
# T3.3 fit_speaker_pca
# ------------------------------------------------------------------------------------------------
def test_fit_speaker_pca_matches_svd():
    x = _pca_data()
    pca = SP.fit_speaker_pca(x, dim=DIM, fit_corpus="toy")
    comp = pca.components
    assert comp.shape == (DIM, D) and pca.mean.shape == (D,) and pca.scales.shape == (DIM,)
    np.testing.assert_allclose(pca.mean, x.mean(0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(comp @ comp.T, np.eye(DIM), rtol=0, atol=1e-10)
    xc = x - x.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    for i in range(DIM):
        # the same axis up to sign
        assert abs(abs(float(comp[i] @ vt[i])) - 1.0) < 1e-9, i
    # the sign rule: the largest-magnitude loading of every component is positive
    rows = np.arange(DIM)
    assert np.all(comp[rows, np.abs(comp).argmax(1)] > 0)
    # the oracle with the sign rule applied is the fitted matrix
    oracle = vt[:DIM] * np.sign(vt[:DIM][rows, np.abs(vt[:DIM]).argmax(1)])[:, None]
    np.testing.assert_allclose(comp, oracle, rtol=0, atol=1e-9)
    # scales = per-component std of the projected fit data; explained variance = s^2 / sum s^2
    np.testing.assert_allclose(pca.scales, (xc @ oracle.T).std(0), rtol=1e-10)
    np.testing.assert_allclose(pca.explained_variance_ratio, (s[:DIM] ** 2) / (s ** 2).sum(), rtol=1e-9)
    assert pca.n_fit_utts == N and pca.fit_corpus == "toy"


def test_fit_speaker_pca_signs_are_deterministic():
    """The sign rule fixes the eigensolver's arbitrary signs: row order and a global sign flip of
    the data (same covariance) give the same components."""
    x = _pca_data(1)
    a = SP.fit_speaker_pca(x, dim=DIM)
    b = SP.fit_speaker_pca(x[::-1].copy(), dim=DIM)
    c = SP.fit_speaker_pca(-x, dim=DIM)
    np.testing.assert_allclose(a.components, b.components, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a.components, c.components, rtol=0, atol=1e-9)


def test_speaker_pca_apply_and_round_trip(tmp_path):
    x = _pca_data(2)
    pca = SP.fit_speaker_pca(x, dim=DIM, fit_corpus="toy")
    eta = pca.apply(x)
    assert eta.dtype == np.float32 and eta.shape == (N, DIM)
    np.testing.assert_allclose(eta.astype(np.float64).mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(eta.astype(np.float64).std(0), 1.0, rtol=1e-5)
    # decorrelated axes
    cov = np.cov(eta.astype(np.float64).T, bias=True)
    np.testing.assert_allclose(cov, np.eye(DIM), atol=1e-5)
    path = str(tmp_path / "pca.npz")
    pca.save(path)
    back = SP.SpeakerPca.load(path)
    for k in ("mean", "components", "scales", "explained_variance_ratio"):
        np.testing.assert_array_equal(getattr(back, k), getattr(pca, k))
    assert back.fit_corpus == "toy" and back.n_fit_utts == N
    np.testing.assert_array_equal(back.apply(x[:7]), pca.apply(x[:7]))
    with pytest.raises(AssertionError):
        pca.apply(x[:, :-1])


# ------------------------------------------------------------------------------------------------
# T3.3 SpeakerEtaJob
# ------------------------------------------------------------------------------------------------
def _stats(mean, rng):
    return np.stack([mean, np.abs(rng.randn(len(mean))) + 0.1]).astype(np.float32)


def _eta_bed(tmp_path, *, leak_dev_into_fit=False):
    rng = np.random.RandomState(3)
    x = _pca_data(3, n=40).astype(np.float32)
    fit_tags = [f"{19 + i % 5}-198-{i:04d}" for i in range(40)]
    # two fit shards, tags in NON-sorted order inside each
    order = rng.permutation(40)
    shards = [{fit_tags[i]: _stats(x[i], rng) for i in order[:25]}, {fit_tags[i]: _stats(x[i], rng) for i in order[25:]}]
    dev_tags = {"dev-clean": ["84-121123-0000", "84-121123-0001"], "dev-other": ["116-288045-0000", "116-288045-0003"]}
    dev_x = rng.randn(5, D).astype(np.float32) * 4
    dev_store = {t: _stats(dev_x[i], rng) for i, t in enumerate(dev_tags["dev-clean"] + dev_tags["dev-other"])}
    dev_store["700-1-0000"] = _stats(dev_x[4], rng)  # in the dev dump, not in gold: must be ignored
    if leak_dev_into_fit:
        shards[0]["116-288045-0003"] = _stats(dev_x[3], rng)
    paths = []
    for k, sh in enumerate(shards):
        p = tmp_path / f"fit{k}.pkl"
        p.write_bytes(pickle.dumps(sh))
        paths.append(tk.Path(str(p)))
    (tmp_path / "dev.pkl").write_bytes(pickle.dumps(dev_store))
    (tmp_path / "gold.json").write_text(json.dumps({s: {t: "AH" for t in ts} for s, ts in dev_tags.items()}))
    n_fit = 40 + int(leak_dev_into_fit)
    job = SP.SpeakerEtaJob(fit_perutt=paths, dev_perutt=[tk.Path(str(tmp_path / "dev.pkl"))],
                           dev_tags_json=tk.Path(str(tmp_path / "gold.json")), dim=DIM, fit_corpus="toy",
                           expect_fit_utts=n_fit, expect_dev_utts=4)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("out_eta", "out_pca", "out_stats", "out_json"):
        fname = {"out_eta": "eta.npz", "out_pca": "speaker_pca.npz", "out_stats": "eta.stats.txt",
                 "out_json": "eta.json"}[name]
        setattr(job, name, tk.Path(str(out / fname)))
    means = {t: np.asarray(v[0], dtype=np.float64) for sh in shards for t, v in sh.items()}
    dev_means = {t: np.asarray(dev_store[t][0], dtype=np.float64) for ts in dev_tags.values() for t in ts}
    return job, out, means, dev_means


def test_speaker_eta_job(tmp_path):
    job, out, fit_means, dev_means = _eta_bed(tmp_path)
    job.run()
    d = np.load(out / "eta.npz", allow_pickle=False)
    tags = [str(t) for t in d["tags"]]
    eta = d["eta"]
    assert tags == sorted(tags) and len(set(tags)) == len(tags)
    assert set(tags) == set(fit_means) | set(dev_means) and "700-1-0000" not in tags
    assert eta.dtype == np.float32 and eta.shape == (len(tags), DIM)
    fit_names = sorted(fit_means)
    oracle = SP.fit_speaker_pca(np.stack([fit_means[t] for t in fit_names]), dim=DIM, fit_corpus="toy")
    all_means = {**fit_means, **dev_means}
    np.testing.assert_allclose(eta, oracle.apply(np.stack([all_means[t] for t in tags])), rtol=0, atol=1e-6)
    saved = SP.SpeakerPca.load(str(out / "speaker_pca.npz"))
    np.testing.assert_allclose(saved.components, oracle.components, rtol=0, atol=1e-12)
    rec = json.load(open(out / "eta.json"))
    assert rec["n_fit_utts"] == 40 and rec["n_dev_utts"] == 4 and rec["n_projected_utts"] == 44 and rec["dim"] == DIM


def test_speaker_eta_job_refuses_dev_in_fit(tmp_path):
    job, _, _, _ = _eta_bed(tmp_path, leak_dev_into_fit=True)
    with pytest.raises(AssertionError, match="fit corpus contains dev"):
        job.run()


def test_speaker_eta_job_count_guard(tmp_path):
    job, _, _, _ = _eta_bed(tmp_path)
    job.expect_fit_utts = 41
    with pytest.raises(AssertionError, match="fit corpus is 40 utts"):
        job.run()


# ------------------------------------------------------------------------------------------------
# T3.4 the eta lookup inside the model and the train step
# ------------------------------------------------------------------------------------------------
B, ETA_DIM, N_UNITS = 2, 4, 20
LENS = [21, 17]
TAGS = ["utt0", "utt1", "utt2"]


@pytest.fixture(scope="module")
def model():
    from returnn.frontend._backend import select_backend_torch

    from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM
    from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior
    from i6_experiments.users.wu.experiments.unsupervised_asr.phones import N_TYPES

    select_backend_torch()
    import tempfile

    tmp = tempfile.mkdtemp(prefix="t34_")
    rng = np.random.RandomState(0)
    n = N_TYPES
    prior = PhoneNgramPrior(np.log(np.ones(n) / n), np.log(rng.dirichlet(np.ones(n), size=n + 1)),
                            np.log(rng.dirichlet(np.ones(n), size=(n + 1) * (n + 1))), meta={"test": True})
    prior.save(f"{tmp}/prior.npz")
    # rows deliberately far apart so a row mix-up cannot hide
    eta = (rng.randn(len(TAGS), ETA_DIM) * 3 + np.arange(len(TAGS))[:, None] * 5).astype(np.float32)
    np.savez_compressed(f"{tmp}/eta.npz", tags=np.array(TAGS), eta=eta)
    m = BM.get_model(
        epoch=1, step=0, device="cpu",
        temperature_schedule=[8.0, 2.0, 2.0], anchor_weight_schedule=0.0, lam_agg=0.1,
        count_ema_decay=0.99, band=4, prior_weight=1.0, prior_npz_path=f"{tmp}/prior.npz",
        eta_table_path=f"{tmp}/eta.npz",
        reverse_kwargs={"n_units": N_UNITS, "d_max": 4, "d_max_sil": 6, "eta_dim": ETA_DIM, "d_model": 16, "d_ff": 16},
        lam_rate=3.0, rate_rho_hz=9.6619373279, rate_fd_eps=0.25, rate_fd_mode="central",
        lattice_reduction="matmul", lattice_checkpoint=2, lattice_float64=True,
    )
    m.eval()
    return m, eta


def test_lookup_eta_rows_in_tag_order(model):
    m, eta = model
    got = m.lookup_eta(["utt2", "utt0", "utt2", "utt1"], "cpu")
    assert got.shape == (4, ETA_DIM) and got.dtype == torch.float32
    np.testing.assert_array_equal(got.numpy(), eta[[2, 0, 2, 1]])
    with pytest.raises(KeyError, match="not in the frozen eta table"):
        m.lookup_eta(["utt0", "nope-1-0000"], "cpu")


class _Stop(Exception):
    pass


def _extern_data(units, tags):
    from returnn.tensor import Dim, Tensor, TensorDict

    g = torch.Generator().manual_seed(0)
    batch = Dim(B, name="batch")
    lens = torch.tensor(LENS, dtype=torch.int32)
    tdim = Dim(name="time", dimension=None, dyn_size_ext=Tensor("t", raw_tensor=lens, dims=[batch], dtype="int32"))
    sdim = Dim(name="units_time", dimension=None,
               dyn_size_ext=Tensor("s", raw_tensor=lens.clone(), dims=[batch], dtype="int32"))
    feat = Dim(1024, name="feat")
    data = TensorDict()
    data.data["features"] = Tensor("features", dims=[batch, tdim, feat], dtype="float16", feature_dim=feat,
                                   raw_tensor=torch.randn(B, max(LENS), 1024, generator=g).half())
    data.data["units"] = Tensor("units", dims=[batch, sdim], dtype="int32", sparse_dim=Dim(N_UNITS, name="u"),
                                raw_tensor=units)
    data.data["original_length"] = Tensor("original_length", dims=[batch, Dim(1, name="o")], dtype="int32",
                                          raw_tensor=torch.tensor([[LENS[0] + 5], [LENS[1]]], dtype=torch.int32))
    tag = Tensor("seq_tag", dims=[batch], dtype="string")
    tag.raw_tensor = np.array(tags)
    data.data["seq_tag"] = tag
    return data


def _seg_in_step(m, units, tags, monkeypatch):
    """Run ``train_step`` up to its ``build_segment_table`` call and return (eta, table) it used."""
    import returnn.frontend as rf

    from i6_experiments.users.wu.experiments.unsupervised_asr.model import train_step as TS

    from i6_experiments.users.wu.experiments.unsupervised_asr.model.lattice import build_segment_table as real

    seen = {}
    assert TS.build_segment_table is real or TS.build_segment_table.__name__ == "spy"

    def spy(reverse_model, units_, eta_, **kw):
        seen["eta"] = eta_.detach().clone()
        seen["seg"] = real(reverse_model, units_, eta_, **kw).detach().clone()
        raise _Stop

    monkeypatch.setattr(TS, "build_segment_table", spy)
    rf.init_train_step_run_ctx(train_flag=False, step=0, epoch=1)
    with pytest.raises(_Stop):
        TS.train_step(model=m, extern_data=_extern_data(units, tags))
    return seen["eta"], seen["seg"]


def test_train_step_segment_table_follows_the_tag(model, monkeypatch):
    from i6_experiments.users.wu.experiments.unsupervised_asr.model.lattice import build_segment_table

    m, eta = model
    g = torch.Generator().manual_seed(1)
    units = torch.randint(0, N_UNITS, (B, max(LENS)), generator=g, dtype=torch.int32)
    with torch.no_grad():
        eta_a, seg_a = _seg_in_step(m, units, ["utt0", "utt2"], monkeypatch)
        # the SAME units under swapped tags: each row's table changes to the other tag's eta
        eta_b, seg_b = _seg_in_step(m, units, ["utt2", "utt0"], monkeypatch)
        # the whole batch permuted (units rows and tags together): the tables permute with it
        eta_c, seg_c = _seg_in_step(m, units.flip(0).contiguous(), ["utt2", "utt0"], monkeypatch)
        np.testing.assert_array_equal(eta_a.numpy(), eta[[0, 2]])
        np.testing.assert_array_equal(eta_b.numpy(), eta[[2, 0]])
        np.testing.assert_array_equal(eta_c.numpy(), eta[[2, 0]])
        torch.testing.assert_close(seg_c, seg_a.flip(0), rtol=1e-5, atol=1e-6)
        # each row is build_segment_table of its own units and its own tag's eta, alone
        for tags, u, seg in ((["utt0", "utt2"], units, seg_a), (["utt2", "utt0"], units, seg_b)):
            for r in range(B):
                one = build_segment_table(m.reverse, u[r:r + 1].long(), torch.as_tensor(eta[[TAGS.index(tags[r])]]))
                torch.testing.assert_close(seg[r:r + 1], one, rtol=1e-5, atol=1e-6)  # float32 convention (plan §0)
        # and the swap did move the tables (the tag, not the position, selects eta)
        assert (seg_a - seg_b).abs().max() > 1e-4
