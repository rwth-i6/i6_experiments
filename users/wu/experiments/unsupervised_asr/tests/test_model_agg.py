"""Oracle tests of the corpus-statistics term L_agg on the blank-free recognizer (test plan
2026-09-24, T1.16 and T1.17; suspicious item S9).

The oracle for the expected counts enumerates every frame path a in K^T of each utterance, collapses
it into its runs (the blank-free B(a)) and counts, with P(a) = prod_t q_t(a_t):

    uni[k]    = E[number of runs of k]
    bi[j, k]  = E[number of adjacent run pairs (j, k)]      (j != k by construction of runs)

It is a float64 torch expression, so autograd of the oracle gives the exact gradient.
"""

import itertools

import numpy as np
import pytest
import torch

from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree as BF
from i6_experiments.users.wu.experiments.unsupervised_asr.model.agg import AggConfig
from i6_experiments.users.wu.experiments.unsupervised_asr.model.blankfree_model import BlankfreeAggLoss

LENS = [5, 3, 1]
T_MAX = 5


def _runs(path):
    return [k for i, k in enumerate(path) if i == 0 or path[i - 1] != k]


def _oracle_counts(log_q: torch.Tensor, lens):
    """``(uni [B, K], bi [B, K, K])`` by enumerating every path of each utterance's first len frames."""
    b, _, k = log_q.shape
    unis, bis = [], []
    for i in range(b):
        n = int(lens[i])
        paths = list(itertools.product(range(k), repeat=n))
        c_uni = np.zeros((len(paths), k))
        c_bi = np.zeros((len(paths), k, k))
        for p, a in enumerate(paths):
            runs = _runs(a)
            for r in runs:
                c_uni[p, r] += 1
            for j, kk in zip(runs[:-1], runs[1:]):
                c_bi[p, j, kk] += 1
        idx = torch.tensor(paths, dtype=torch.long)  # [N, n]
        frames = torch.arange(n).view(1, -1).expand_as(idx)
        log_p = log_q[i][frames, idx].sum(dim=1)  # [N]
        prob = log_p.exp()
        unis.append(prob @ torch.as_tensor(c_uni, dtype=log_q.dtype))
        bis.append(torch.einsum("n,njk->jk", prob, torch.as_tensor(c_bi, dtype=log_q.dtype)))
    return torch.stack(unis), torch.stack(bis)


def _logits(k, seed, b=3, t=T_MAX):
    g = torch.Generator().manual_seed(seed)
    return (1.5 * torch.randn(b, t, k, generator=g, dtype=torch.float64)).requires_grad_(True)


# --- T1.16 -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 4])
def test_expected_run_counts_match_enumeration(k):
    """T1.16: values (1e-12), zero diagonal, padding contributes nothing, gradients of a random
    linear functional (1e-10)."""
    lens = torch.tensor(LENS)
    logits = _logits(k, seed=k)
    log_q = torch.log_softmax(logits, dim=-1)
    uni, bi = BF.expected_run_counts(log_q, lens)
    o_uni, o_bi = _oracle_counts(log_q, LENS)
    assert uni.shape == (3, k) and bi.shape == (3, k, k)
    assert float((uni - o_uni).abs().max()) <= 1e-12
    assert float((bi - o_bi).abs().max()) <= 1e-12
    assert bool((torch.diagonal(bi, dim1=1, dim2=2) == 0).all())
    # the one-frame utterance has exactly one run and no pair
    assert abs(float(uni[2].sum()) - 1.0) <= 1e-12 and float(bi[2].abs().sum()) == 0.0

    g = torch.Generator().manual_seed(100 + k)
    w_uni = torch.randn(3, k, generator=g, dtype=torch.float64)
    w_bi = torch.randn(3, k, k, generator=g, dtype=torch.float64)
    (grad,) = torch.autograd.grad((w_uni * uni).sum() + (w_bi * bi).sum(), logits, retain_graph=True)
    (o_grad,) = torch.autograd.grad((w_uni * o_uni).sum() + (w_bi * o_bi).sum(), logits)
    assert float((grad - o_grad).abs().max()) <= 1e-10
    pad = torch.arange(T_MAX).view(1, -1) >= lens.view(-1, 1)
    assert float(grad[pad].abs().max()) == 0.0

    # padding: any value past the end changes nothing
    noisy = log_q.detach().clone()
    noisy[pad] = torch.log_softmax(torch.randn(int(pad.sum()), k, dtype=torch.float64), dim=-1)
    uni2, bi2 = BF.expected_run_counts(noisy, lens)
    assert torch.equal(uni2, uni.detach()) and torch.equal(bi2, bi.detach())


# --- T1.17 -------------------------------------------------------------------------------------------

K_AGG, DECAY = 4, 0.99


def _agg_module():
    g = torch.Generator().manual_seed(7)
    text_uni = torch.rand(K_AGG, generator=g) + 0.1
    joint = torch.rand(K_AGG, K_AGG, generator=g) + 0.1
    text_bi = BF.project_text_bigram(joint / joint.sum())
    return BlankfreeAggLoss(AggConfig(count_ema_decay=DECAY, n_phones=K_AGG), text_uni, text_bi)


def _kl(text: torch.Tensor, hat: torch.Tensor) -> torch.Tensor:
    """sum over cells with text > 0 of text (log text - log hat), 0 log 0 = 0."""
    m = text > 0
    return (text[m] * (text[m].log() - hat[m].log())).sum()


def _normalised(log_q, lens):
    o_uni, o_bi = _oracle_counts(log_q, lens)
    u, b = o_uni.sum(0), o_bi.sum(0)
    return u / u.sum(), b / b.sum()


def test_agg_value_ema_and_gradient():
    """T1.17 (a) step-1 value = KL(text || normalised batch counts), uni + bi; (b) three train steps
    follow hat = batch, then d * ema + (1 - d) * batch, and the buffers move only in train mode with
    update_ema; (c) from step 2 the gradient is (1 - d) x the KL's gradient with the EMA constant."""
    agg = _agg_module()
    text_uni, text_bi = agg.text_uni.double(), agg.text_bi.double()  # float32-rounded targets
    lens = torch.tensor(LENS)
    agg.train()
    ema_uni = ema_bi = None
    for step in range(1, 4):
        logits = _logits(K_AGG, seed=200 + step)
        log_q = torch.log_softmax(logits, dim=-1)
        n_uni, n_bi = _normalised(log_q, LENS)
        if step == 1:
            hat_uni, hat_bi = n_uni, n_bi
        else:
            # the module's own (float32) EMA buffers are the constant history
            ema_uni_c, ema_bi_c = agg.ema_uni.double(), agg.ema_bi.double()
            assert torch.allclose(ema_uni_c, ema_uni.detach(), rtol=1e-5, atol=0)
            assert torch.allclose(ema_bi_c[text_bi > 0], ema_bi.detach()[text_bi > 0], rtol=1e-5, atol=0)
            hat_uni = DECAY * ema_uni_c + (1 - DECAY) * n_uni
            hat_bi = DECAY * ema_bi_c + (1 - DECAY) * n_bi
        kl_uni, kl_bi = _kl(text_uni, hat_uni), _kl(text_bi, hat_bi)

        # (b) eval mode and update_ema=False leave every buffer alone
        before = {n: b.clone() for n, b in agg.named_buffers()}
        agg.eval()
        agg(log_q.detach(), lens)
        agg.train()
        agg(log_q.detach(), lens, update_ema=False)
        for n, b in agg.named_buffers():
            assert torch.equal(b, before[n]), n

        loss, stats = agg(log_q, lens)
        expect = float(kl_uni + kl_bi)
        assert abs(float(loss) - expect) <= 1e-10 * max(1.0, abs(expect)), (step, float(loss), expect)
        assert abs(stats["agg/kl_unigram"] - float(kl_uni)) <= 1e-10
        assert abs(stats["agg/kl_bigram"] - float(kl_bi)) <= 1e-10
        assert int(agg.ema_steps) == step

        # (c) gradient: at step 1 the full KL gradient, from step 2 (1 - d) x d KL / d hat . d batch
        (grad,) = torch.autograd.grad(loss, logits, retain_graph=True)
        v_uni = (-text_uni / hat_uni).detach()  # d KL / d hat = -text / hat
        v_bi = torch.where(text_bi > 0, -text_bi / hat_bi.clamp(min=1e-300), torch.zeros_like(text_bi)).detach()
        factor = 1.0 if step == 1 else (1.0 - DECAY)
        (o_grad,) = torch.autograd.grad(factor * ((v_uni * n_uni).sum() + (v_bi * n_bi).sum()), logits)
        scale = float(o_grad.abs().max())
        assert float((grad - o_grad).abs().max()) <= 1e-9 * scale, (step, float((grad - o_grad).abs().max()), scale)

        # the hand recursion in float64
        ema_uni = hat_uni if step == 1 else DECAY * ema_uni + (1 - DECAY) * n_uni
        ema_bi = hat_bi if step == 1 else DECAY * ema_bi + (1 - DECAY) * n_bi
    assert torch.allclose(agg.ema_uni.double(), ema_uni.detach(), rtol=1e-5, atol=0)


# --- T1.17 (d): the targets the bed builds ---------------------------------------------------------


@pytest.fixture(scope="module")
def bed_model(tmp_path_factory):
    """SaeBlankfreeModelV1 built as the bed builds it, on a Witten-Bell prior fitted to a tiny corpus
    with repeated phones and SIL runs (so the text's own diagonal is non-zero)."""
    from returnn.frontend._backend import select_backend_torch

    from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_prior as P
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM

    select_backend_torch()
    tmp = tmp_path_factory.mktemp("agg_bed")
    corpus = (["<SIL> AA B <SIL>", "<SIL> AA B AA <SIL>", "B IY", "<SIL> IY IY B <SIL>"] * 30
              + ["AA", "AA AA B", "<SIL> <SIL> AA", "B IY IY IY"] * 7)
    counts, _ = P.count_ngrams([l.split() for l in corpus])
    prior = P.PhoneNgramPrior.from_counts(counts)
    prior_path = str(tmp / "prior.npz")
    prior.save(prior_path)
    eta_path = str(tmp / "eta.npz")
    np.savez_compressed(eta_path, tags=np.array(["u0"]), eta=np.zeros((1, 4), dtype=np.float32))
    # construction arguments as in test_model_blankfree._kwargs; only the agg buffers are read
    model = BM.SaeBlankfreeModelV1(
        temperature_schedule=[2.0], anchor_weight_schedule=0.0, lam_agg=0.1, count_ema_decay=0.99,
        band=4, prior_weight=1.0, prior_npz_path=prior_path, eta_table_path=eta_path,
        reverse_kwargs={"n_units": 20, "d_max": 4, "d_max_sil": 6, "eta_dim": 4, "d_model": 16, "d_ff": 16},
        lam_rate=3.0, rate_rho_hz=9.6619373279, rate_fd_eps=0.25, rate_fd_mode="central",
        lattice_reduction="matmul", lattice_checkpoint=2, lattice_float64=True,
    )
    return model, P.PhoneNgramPrior.load(prior_path)


def test_bed_agg_targets(bed_model):
    """T1.17 (d): text_uni = unigram_probs() renormalised over the 40 symbols; text_bi =
    offdiag(uni[h] exp(log_bi[h, k])) / sum with the BOS row dropped and the diagonal exactly 0.
    RECORD: the unigram target is not projected onto the run-collapse support."""
    model, prior = bed_model
    agg = model.agg
    assert isinstance(agg, BlankfreeAggLoss)
    k = 40
    uni = np.exp(prior.log_uni.astype(np.float64))
    uni = uni / uni.sum()
    joint = uni[:, None] * np.exp(prior.log_bi[:k, :k])  # rows h = 0..39; BOS row 40 dropped
    joint_full = joint / joint.sum()
    off = joint.copy()
    np.fill_diagonal(off, 0.0)
    off = off / off.sum()
    got_uni = agg.text_uni.double().numpy()
    got_bi = agg.text_bi.double().numpy()
    assert agg.text_uni.dtype == torch.float32 and agg.text_bi.dtype == torch.float32
    assert np.abs(got_uni - uni).max() <= 1e-5 * uni.max()
    assert np.all(np.abs(got_uni - uni) <= 1e-5 * uni)  # float32 1e-5 relative, cell by cell
    assert np.all(np.abs(got_bi - off) <= 1e-5 * off + 1e-12)
    assert np.all(np.diag(got_bi) == 0.0)

    # RECORD (S9): the bigram target was projected onto adjacent-distinct pairs, the unigram was not.
    diag_mass = float(np.trace(joint_full))
    # the unigram of RUNS under the text's own bigram chain: a k-token starts a new run with
    # probability 1 - p(k | k), so the run-collapse unigram is proportional to uni[k] (1 - p(k|k))
    p_stay = np.exp(np.diag(prior.log_bi[:k, :k]))
    run_uni = uni * (1.0 - p_stay)
    run_uni = run_uni / run_uni.sum()
    l1 = float(np.abs(uni - run_uni).sum())
    kl = float(np.sum(uni * (np.log(uni) - np.log(run_uni))))
    print(f"\nT1.17 RECORD (toy WB prior): diagonal mass removed from the bigram target = {diag_mass:.6f}; "
          f"unigram target vs run-collapse unigram uni*(1-p(k|k)): L1 = {l1:.6f}, KL = {kl:.6f} nats; "
          f"projected-bigram row marginal vs text_uni L1 = {float(np.abs(got_bi.sum(1) - got_uni).sum()):.6f}")
    assert 0.0 < diag_mass < 1.0 and np.isfinite(l1) and np.isfinite(kl) and kl >= 0.0
