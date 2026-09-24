"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_genmarg_jobs.py (the RETURNN side:
phi loading, ``genmarg_get_model``, the computations, the two forward steps and their callbacks).

The label-free generative reads of L2-0 / L2-1, run inside RETURNN by the ``ReturnnForwardJobV2``
that ``genmarg.genmarg_reads`` builds:

* :func:`genmarg_get_model` -- ``SaeBlankfreeModelV1`` from the bed's model args (both init paths
  and every k2 key refused) with phi loaded STRICTLY into ``model.reverse`` from a reverse-only
  checkpoint (unprefixed keys) or a whole-model training checkpoint (its ``reverse.*`` keys), every
  live reverse tensor checked with ``torch.equal``.  The recognizer is never read.
* :func:`genmarg_forward_step` + :func:`GenMargCallback` -- statistic (a): per utterance the tau = 1
  generative marginal, the tau = 2 free energy and the expected non-SIL rate (``genmarg.json``).
* :func:`gendecode_forward_step` + :func:`GenDecodeCallback` -- phi's tau = 1 generative posterior
  decode, validity-checked (``decode_raw.json``, ``gendecode.json`` with the EMITTED non-SIL rate
  every L2-1 rate clause reads).

Port: code verbatim; imports point at this package (``model.blankfree_model``, ``model.lattice``,
``model.rate_term``, ``model.blankfree_permute``, ``model.train_step``, ``phones``) and at
``genmarg_decode`` for the Viterbi pass and the per-segment re-add (``soft_scorer`` in the source).
Statistic (b) (``GenerativeDecodeGapJob``) and the label-using report are not ported
(``genmarg.genmarg_reads`` refuses ``gap`` / ``report``); the conventions below are the source's
text unchanged, because the callbacks stamp them into every json.

CONVENTIONS (registered here, with the producing code)

* NULL RECOGNIZER.  ``log_q`` is IDENTICALLY ZERO (float64) on every recognizer frame and symbol, so
  every recognizer arc weight is 0 and the lattice's log Z is exactly
  ``log sum_a exp((log P_psi(B(a)) + log p_phi(z | B(a), eta)) / tau)`` -- the formula of L2-0 (a),
  with the lattice's own path set a.  A UNIFORM recognizer (q = 1/40, what zero logits give after the
  log-softmax, i.e. the L2-1 training objective's null theta) differs by the per-utterance constant
  ``(T_rec / tau) log 40`` with ``T_rec = ceil(S / 3)`` recognizer frames: that constant is REMOVED
  here and reported per utterance (``uniform_q_offset_tau1``, ``uniform_q_offset_tau2``) so the
  uniform-q value is ``value + offset``.  It depends on the utterance length only, so it moves no
  comparison between two phis on the same utterances, nor between a real and a within-utterance
  shuffled stream (same lengths).
* THE LATTICE.  The bed's own: ``SaeBlankfreeModelV1``'s ``lattice_cfg`` (blank-free topology,
  recognizer stride 3, band 25, d_min 2, D 25 / D_sil 50), its trigram ``prior_history`` and
  ``prior_log_bi`` from the bed's prior (``PhoneNgramPriorJob.RtzbESkOedsT``) at the model's
  ``prior_weight`` (1.0 in the bed; asserted to be a scalar, not a schedule), anchor weight 0, the
  model's ``lattice_reduction``; segment table ``lattice.build_segment_table`` (phi's emission +
  duration heads) in FLOAT64, as the bed's ``lattice_float64``.  The l_tau lattice is EXACT (no
  pruning); max_active exists only in the k2 leg, which is not computed here (next item).
* THE k2 WORD-GRAPH TERM IS NOT COMPUTED (amendment A1: dropped).  ``LexlatK2Runtime.step`` takes
  the recognizer's emissions ``log_q`` alone (``L_lex = log Z_HLG(e / tau) - log Z_H(e / tau)``,
  e = log_q); phi does not enter it.  Under the null recognizer it is therefore the same number for
  every phi on a given utterance, and a total "l_tau + k2" ranks phis exactly as l_tau does.
* VALUES, per utterance (S = retained unit frames, the l_tau divisor of the train step), with
  Z_tau = sum_a exp((log P_psi + log p_phi) / tau) and the free energy F_tau = -tau log Z_tau:
  ``nll_tau1 = -log Z_1 = F_1`` (the tau = 1 negative log marginal likelihood, nats);
  ``free_energy_tau2 = F_2 = -2 log Z_2`` -- THE QUANTITY BANKED AT tau = 2 (statistic (a)'s
  "tau = 2 free energy", on the tau = 1 nats scale);
  ``l_tau2 = -log Z_2 = F_2 / 2`` -- what the DP / ``lattice_loss`` returns at tau = 2
  (``lattice.py``: "L_tau = F_tau / tau"), kept only for cross-reference with training logs;
  each also divided by S (``*_per_frame``).  LOWER IS BETTER for every value.
* EMITTED RATE (amendment A8; THE rate of every L2-1 rate clause: A2's VOID, A3's probe PASS, NO
  ELIGIBLE RESTART).  A8, verbatim: "[5.80, 14.49] Hz is a greedy emitted rate per original second,
  \"never a posterior expectation\" [...] Under the null recognizer the emitted sequence is phi's
  decode of the tau = 1 generative posterior (the genmarg decode). Rate = 50 x (sum of non-SIL
  tokens in that decode) / (sum of original 50 Hz frames), pooled over the CV holdout".  Per
  decoded utterance (``gendecode.json``): ``emitted_nonsil_tokens`` = the decode's segments whose
  symbol is not SIL, ``original_frames`` = its ``orig_length``, ``emitted_nonsil_rate_original_hz =
  50 * tokens / original_frames``; summary ``emitted_nonsil_rate``: ``pooled_hz = 50 * sum tokens /
  sum original_frames`` over every decoded (non-impossible) utterance of the set (the gated value)
  and ``utterance_mean_hz`` (reported).
* EXPECTED RATE (REPORT ONLY since amendment A8; enters no clause).  Under the tau = 1 generative
  posterior (``lattice.lattice_forward_backward`` at tau = 1, null recognizer, same lattice; its
  log Z_1 is asserted equal to the forward pass's), ``expected_nonsil_tokens`` =
  ``rate_term.expected_nonsil_tokens`` (seg_post summed over the non-SIL types; SIL excluded);
  ``expected_nonsil_rate_original_hz = 50 * E[N_nonSIL] / orig_length`` (the ORIGINAL-audio
  denominator of the train step's rate and of G4a.9's ``phone_rate_original_hz``, which the
  [5.80, 14.49] Hz band was read on) and ``expected_nonsil_rate_retained_hz = 50 * E[N_nonSIL] / S``
  (reported).  Summary: ``pooled_hz = 50 * sum E / sum orig_length`` and ``utterance_mean_hz``,
  both over the finite utterances.  A2 read this; A8 replaced it by the emitted rate above, and it
  is kept in the jsons as a report-only column.
* IMPOSSIBLE UTTERANCES.  An utterance whose log Z is NEG_INF-floored (``<= NEG_INF / 2``) or not
  finite has no path: its values are ``null`` in the json, it is COUNTED and LISTED separately and
  it never enters a mean.
* MEANS.  ``utterance_mean_per_frame`` = mean over the finite utterances of the per-utterance
  per-frame value (the train step's l_tau reduction); ``pooled_per_frame`` = sum of values / sum of
  S over the same utterances.  The selection reader uses ``utterance_mean_per_frame`` over the tags
  finite in EVERY gated json.
* SHUFFLE.  ``shuffle_seed`` (None = the real stream) permutes the unit stream within each
  utterance with ``blankfree_permute.permute_frames`` -- ONE permutation per utterance drawn from
  (seed, tag) alone, identity on padding, the permutation the bed's ``permute_frames_seed`` applies
  -- so a null restart fitted on a corpus shuffled with seed s is scored on the held-out stream
  shuffled by the same function with the same s.  Lengths and eta are unchanged.
* THE DECODE.  ``soft_scorer.viterbi_blankfree`` at tau = 1 under the null recognizer: the single
  max-weight path (the mode of the generative posterior).  Its symbol string (emitted tokens in
  frame order, SIL kept, phone names ``prior.PHONES``) is written as ``decode_raw.json``
  ``{tag: [phone, ...]}`` -- the decode column format of ``BlankfreeDecodeGapJob`` /
  ``greedy_raw.json``.  Per live utterance the job ASSERTS: the path weight re-added from its own
  per-token terms (``soft_scorer.segment_conditionals``) equals the max-plus maximum; that maximum is
  at most log Z_1; the segments tile [0, S); every duration is in [d_min, D_k]; the unit position
  after every recognizer frame t is within the band, |s - 3 (t + 1)| <= W.
* STATISTIC (b) (:class:`GenerativeDecodeGapJob`).  ``reverse.evaluate`` (D10b's scorer) scores the
  utterance's units under phi with its OWN decode and with the decode of a same-speaker donor
  (``s1a_job.build_derangement`` over the decodes: nearest token count, donor string must cover the
  utterance's frames); per utterance ``own - deranged`` in nats, per frame and per own-decode token,
  with a speaker-clustered bootstrap CI (``d8_admission.cluster_bootstrap``, the campaign's
  ``PAIRED_PER_*`` resamples / seed / clustering); positive = phi prefers the utterance's own decode.
  An utterance with no admissible donor (a speaker with one utterance in the set) is dropped and
  counted.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "genmarg_get_model", "genmarg_forward_step", "GenMargCallback", "gendecode_forward_step",
    "GenDecodeCallback", "generative_marginals", "generative_decode", "load_phi_state",
    "marginal_rows", "summarise_marginals", "emitted_rate_summary", "rate_summary",
]

#: model args the eval refuses: an init path would load a submodule before phi's own strict load;
#: a k2 key would build the k2 runtime, which this read does not compute (module docstring)
INIT_PATH_KEYS = ("recognizer_checkpoint_path", "reverse_checkpoint_path")
K2_KEY_PREFIX = "lexlat_k2_"
REVERSE_PREFIX = "reverse."
#: the two temperatures of statistic (a) (L2-0 "Competence statistics": tau = 1 and the tau = 2
#: free energy)
TEMPERATURES = (1.0, 2.0)
#: amendment A2 / A8: the bed's rate band, Hz, a greedy EMITTED non-SIL rate per original second
#: (``SAE_4A_attrib.md`` S3b-R); read on the emitted rate of the genmarg decode (A8); outside = VOID
RATE_BAND_HZ = (5.80, 14.49)
FRAME_RATE_HZ = 50.0
#: implementation tolerances (float64 identities, not experimental constants)
_READD_RTOL = 1e-8
_LOGZ_ATOL = 1e-6


# =================================================================================================
# phi loading
# =================================================================================================
def load_phi_state(path: str):
    """``(reverse state dict, meta)`` from a reverse-only or a whole-model RETURNN checkpoint.

    A checkpoint with ANY ``reverse.*`` key is a whole-model checkpoint and exactly its ``reverse.*``
    keys (prefix stripped) are phi; otherwise every key is phi (``BlankfreeSupervisedReverseInitJob``,
    ``ExtractSubmoduleCheckpointJob``).  The caller loads it strictly.
    """
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(ckpt, dict) and "model" in ckpt, f"{path}: not a RETURNN checkpoint"
    sd = ckpt["model"]
    if any(k.startswith(REVERSE_PREFIX) for k in sd):
        state = {k[len(REVERSE_PREFIX):]: v for k, v in sd.items() if k.startswith(REVERSE_PREFIX)}
        fmt = "whole_model"
    else:
        state = dict(sd)
        fmt = "reverse_only"
    meta = {"phi_checkpoint": str(path), "format": fmt, "n_keys": len(state),
            "epoch": int(ckpt.get("epoch", -1)), "step": int(ckpt.get("step", -1))}
    return state, meta


def _check_model_args(model_args: Dict[str, Any]) -> None:
    stated = [k for k in INIT_PATH_KEYS if model_args.get(k)]
    assert not stated, f"the genmarg read loads phi itself; model_args state {stated}"
    k2 = [k for k in model_args if k.startswith(K2_KEY_PREFIX)]
    assert not k2, f"the genmarg read computes no k2 term (module docstring); model_args state {k2}"


# =================================================================================================
# 1. get_model
# =================================================================================================
def genmarg_get_model(*, phi_checkpoint: str, model_args: Dict[str, Any], **kwargs):
    """``SaeBlankfreeModelV1`` (bed args) with phi loaded strictly into ``model.reverse``."""
    import torch

    from ..model.blankfree_model import SaeBlankfreeModelV1

    _check_model_args(model_args)
    model = SaeBlankfreeModelV1(**model_args, **kwargs)
    assert getattr(model, "lexlat_k2", None) is None, "a k2 runtime was built; it is not computed here"
    state, meta = load_phi_state(phi_checkpoint)
    model.reverse.load_state_dict(state, strict=True)
    for key, value in model.reverse.state_dict().items():
        assert torch.equal(value.detach().cpu(), state[key].detach().cpu()), f"reverse.{key} differs"
    model._genmarg_load = dict(meta, strict=True)
    model._genmarg_prior = model_args.get("prior_npz_path")
    model._genmarg_batches = []
    print(f"genmarg load: {json.dumps(model._genmarg_load)}", flush=True)
    return model


# =================================================================================================
# 2. the computations, importable and testable without RETURNN
# =================================================================================================
def _setup(model, units, unit_lens, tags, shuffle_seed):
    """``(units, dp)``: the (optionally shuffled) unit stream and the lattice inputs, float64."""
    import torch

    from ..model.blankfree_permute import permute_frames
    from ..model.lattice import build_segment_table

    if shuffle_seed is not None:
        (units,) = permute_frames((units,), tags=tags, lengths=unit_lens, seed=int(shuffle_seed))
    prior_weight = model.prior_weight_at(1)
    assert getattr(model, "prior_weight_schedule", None) is None, "a beta schedule is not the bed's"
    cfg = model.lattice_cfg
    assert cfg.topology == "blankfree" and cfg.recognizer_stride == 3, cfg
    eta = model.lookup_eta(tags, units.device)
    seg = build_segment_table(model.reverse, units, eta, detach=True).double()
    feat_lens = model.recognizer.output_lengths(unit_lens)
    t_rec = (int(units.shape[1]) + cfg.recognizer_stride - 1) // cfg.recognizer_stride
    assert int(feat_lens.max()) <= t_rec, (int(feat_lens.max()), t_rec)
    log_q = torch.zeros((int(units.shape[0]), t_rec, cfg.n_symbols), dtype=torch.float64,
                        device=units.device)
    dp = dict(log_q=log_q, seg=seg, prior=model.prior_log_bi.double().to(units.device),
              feat_lens=feat_lens, unit_lens=unit_lens, cfg=cfg, history=model.prior_history,
              reduction=model.lattice_reduction, prior_weight=float(prior_weight),
              checkpoint=int(model.lattice_checkpoint))
    return units, dp


def generative_marginals(model, units, unit_lens, tags, *, shuffle_seed=None,
                         temperatures=TEMPERATURES, rate: bool = True) -> Dict[str, Any]:
    """``{"log_z": {tau: [B] float64}, "t_rec": [B], "units": the scored stream, "dp",
    "expected_nonsil": [B] or None}`` (no grad; module conventions).

    ``rate``: also run the tau = 1 forward-backward for the expected non-SIL token count under the
    generative posterior (its log Z_1 is asserted equal to the forward pass's).
    """
    import torch

    from ..model.lattice import forward_log_z, lattice_forward_backward
    from ..model.rate_term import expected_nonsil_tokens

    with torch.no_grad():
        units, dp = _setup(model, units, unit_lens, tags, shuffle_seed)
        out = {}
        for tau in temperatures:
            out[float(tau)] = forward_log_z(
                dp["log_q"], dp["seg"], dp["prior"], dp["feat_lens"], dp["unit_lens"], dp["cfg"],
                temperature=float(tau), anchor_weight=0.0, prior_weight=dp["prior_weight"],
                history=dp["history"], reduction=dp["reduction"])
        expected = None
        if rate:
            fb = lattice_forward_backward(
                dp["log_q"], dp["seg"], dp["prior"], dp["feat_lens"], dp["unit_lens"], dp["cfg"],
                temperature=1.0, anchor_weight=0.0, prior_weight=dp["prior_weight"],
                collect_stats=True, history=dp["history"], reduction=dp["reduction"],
                checkpoint=dp["checkpoint"])
            live = ~fb.z_zero
            gap = (fb.log_z - out[1.0]).abs()[live]
            scale = out[1.0].abs()[live].clamp(min=1.0)
            assert bool((gap <= _READD_RTOL * scale).all()), (
                "forward-backward and forward log Z_1 disagree", float(gap.max()))
            expected = expected_nonsil_tokens(fb, dp["cfg"]).double()
    return {"log_z": out, "t_rec": dp["feat_lens"], "units": units, "dp": dp,
            "expected_nonsil": expected}


def _impossible(log_z) -> List[bool]:
    from ..model.lattice import NEG_INF

    return [(not math.isfinite(float(v))) or float(v) <= NEG_INF / 2 for v in log_z.tolist()]


RATE_KEYS = ("expected_nonsil_tokens", "expected_nonsil_rate_original_hz",
             "expected_nonsil_rate_retained_hz")


def marginal_rows(res, unit_lens, tags, original_lens=None) -> List[Dict[str, Any]]:
    """Per-utterance rows of statistic (a) and the expected rate (module conventions).

    ``original_lens``: per utterance ``orig_length`` (50 Hz frames of the original audio, the
    rate's denominator); required when ``res`` carries the expected count.
    """
    lz1, lz2 = res["log_z"][1.0], res["log_z"][2.0]
    imp1, imp2 = _impossible(lz1), _impossible(lz2)
    log_v = math.log(res["dp"]["cfg"].n_symbols)
    expected = res.get("expected_nonsil")
    if expected is not None:
        assert original_lens is not None, "the expected rate needs orig_length"
        expected = expected.cpu().tolist()
    rows = []
    for i, tag in enumerate(tags):
        s = int(unit_lens[i])
        t_rec = int(res["t_rec"][i])
        impossible = bool(imp1[i] or imp2[i])
        row = {"tag": tag, "frames": s, "recognizer_frames": t_rec, "impossible": impossible,
               "uniform_q_offset_tau1": t_rec * log_v, "uniform_q_offset_tau2": t_rec * log_v / 2.0}
        if original_lens is not None:
            row["original_frames"] = int(original_lens[i])
            assert row["original_frames"] >= s, (tag, row["original_frames"], s)
        if impossible:
            row.update({k: None for k in ("log_z_tau1", "log_z_tau2", "nll_tau1", "l_tau2",
                                          "free_energy_tau2", "nll_tau1_per_frame",
                                          "l_tau2_per_frame", "free_energy_tau2_per_frame")
                        + RATE_KEYS})
        else:
            z1, z2 = float(lz1[i]), float(lz2[i])
            row.update({"log_z_tau1": z1, "log_z_tau2": z2, "nll_tau1": -z1, "l_tau2": -z2,
                        "free_energy_tau2": -2.0 * z2, "nll_tau1_per_frame": -z1 / s,
                        "l_tau2_per_frame": -z2 / s, "free_energy_tau2_per_frame": -2.0 * z2 / s})
            if expected is not None:
                e = float(expected[i])
                row.update({"expected_nonsil_tokens": e,
                            "expected_nonsil_rate_original_hz":
                                FRAME_RATE_HZ * e / row["original_frames"],
                            "expected_nonsil_rate_retained_hz": FRAME_RATE_HZ * e / s})
        rows.append(row)
    return rows


VALUE_KEYS = ("nll_tau1", "l_tau2", "free_energy_tau2")
#: the one statement of what each banked value is (printed into every json)
VALUE_LABELS = {
    "nll_tau1": "F_1 = -log Z_1, tau = 1 negative log generative marginal likelihood (nats)",
    "free_energy_tau2": "F_2 = -2 log Z_2, THE tau = 2 free energy banked by statistic (a) (nats)",
    "l_tau2": "F_2 / 2 = -log Z_2, the DP / lattice_loss return at tau = 2 (cross-reference only)",
}


def rate_summary(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pooled and utterance-mean expected non-SIL rate over the finite rows (module conventions)."""
    finite = [r for r in rows if not r["impossible"]]
    if not finite or finite[0].get("expected_nonsil_tokens") is None:
        return None
    e = sum(r["expected_nonsil_tokens"] for r in finite)
    orig = sum(r["original_frames"] for r in finite)
    ret = sum(r["frames"] for r in finite)
    return {"n": len(finite), "expected_nonsil_tokens": e, "original_frames": orig,
            "retained_frames": ret,
            "original_hz": {"pooled_hz": FRAME_RATE_HZ * e / orig,
                            "utterance_mean_hz": sum(r["expected_nonsil_rate_original_hz"]
                                                     for r in finite) / len(finite)},
            "retained_hz": {"pooled_hz": FRAME_RATE_HZ * e / ret,
                            "utterance_mean_hz": sum(r["expected_nonsil_rate_retained_hz"]
                                                     for r in finite) / len(finite)},
            "posterior": "tau = 1 generative posterior (null recognizer)",
            "role": ("REPORT ONLY (amendment A8): an expected count; the gated rate is the emitted "
                     "rate of the decode, gendecode.json emitted_nonsil_rate")}


def summarise_marginals(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Means over the finite utterances; the impossible ones counted and listed, never averaged."""
    finite = [r for r in rows if not r["impossible"]]
    out: Dict[str, Any] = {"n_utterances": len(rows), "n_finite": len(finite),
                           "n_impossible": len(rows) - len(finite),
                           "impossible_tags": sorted(r["tag"] for r in rows if r["impossible"]),
                           "value_labels": dict(VALUE_LABELS),
                           "tau2_banked": "free_energy_tau2"}
    frames = sum(r["frames"] for r in finite)
    for key in VALUE_KEYS:
        if finite:
            out[key] = {
                "utterance_mean_per_frame": sum(r[f"{key}_per_frame"] for r in finite) / len(finite),
                "pooled_per_frame": sum(r[key] for r in finite) / frames,
                "utterance_mean": sum(r[key] for r in finite) / len(finite),
            }
        else:
            out[key] = None
    out["frames_finite"] = frames
    out["expected_nonsil_rate"] = rate_summary(rows)
    return out


def generative_decode(model, units, unit_lens, tags, *, shuffle_seed=None,
                      original_lens=None) -> Dict[str, Any]:
    """The tau = 1 max-plus path under the null recognizer, validity-checked (module conventions).

    Returns per-utterance rows ``{tag, frames, impossible, log_w, log_z_tau1, tokens, segments}``
    (``segments`` = ``[[phone id, start, duration], ...]`` in order).  Raises on any failed check.
    ``original_lens`` (per utterance ``orig_length``): also the emitted non-SIL rate per row
    (``original_frames``, ``emitted_nonsil_tokens``, ``emitted_nonsil_rate_original_hz``; A8).
    """
    import torch

    from ..model.lattice import forward_log_z
    from ..phones import PHONES, SIL_ID
    from .genmarg_decode import segment_conditionals, viterbi_blankfree

    with torch.no_grad():
        units, dp = _setup(model, units, unit_lens, tags, shuffle_seed)
        cfg = dp["cfg"]
        path = viterbi_blankfree(
            dp["log_q"], dp["seg"], dp["prior"], dp["feat_lens"], dp["unit_lens"], cfg,
            temperature=1.0, anchor_weight=0.0, prior_weight=dp["prior_weight"],
            history=dp["history"], checkpoint=dp["checkpoint"])
        log_z = forward_log_z(
            dp["log_q"], dp["seg"], dp["prior"], dp["feat_lens"], dp["unit_lens"], cfg,
            temperature=1.0, anchor_weight=0.0, prior_weight=dp["prior_weight"],
            history=dp["history"], reduction=dp["reduction"])
        post = segment_conditionals(
            path, dp["log_q"], dp["seg"], dp["prior"], cfg, temperature=1.0, anchor_weight=0.0,
            prior_weight=dp["prior_weight"], history=dp["history"])
    imp_path = [bool(v) for v in path.z_zero.tolist()]
    imp_z = _impossible(log_z)
    ep = path.emit_phone.cpu().numpy()
    es = path.emit_start.cpu().numpy()
    ed = path.emit_duration.cpu().numpy()
    lw = path.log_w.cpu().tolist()
    readd = post.path_log_w.cpu().tolist()
    lz = log_z.cpu().tolist()
    t_lens = dp["feat_lens"].cpu().tolist()
    stride, w = cfg.recognizer_stride, cfg.band
    rows = []
    for i, tag in enumerate(tags):
        s_len = int(unit_lens[i])
        t_len = int(t_lens[i])
        assert imp_path[i] == imp_z[i], (tag, "max-plus and sum forward disagree on Z = 0")
        row = {"tag": tag, "frames": s_len, "recognizer_frames": t_len, "impossible": imp_path[i]}
        if original_lens is not None:
            row["original_frames"] = int(original_lens[i])
            assert row["original_frames"] >= s_len, (tag, row["original_frames"], s_len)
        if imp_path[i]:
            row.update({"log_w": None, "log_z_tau1": None, "tokens": None, "segments": None})
            if original_lens is not None:
                row.update({"emitted_nonsil_tokens": None, "emitted_nonsil_rate_original_hz": None})
            rows.append(row)
            continue
        # (1) the path's weight re-added from its own per-token terms is the max-plus maximum
        assert abs(readd[i] - lw[i]) <= _READD_RTOL * max(1.0, abs(lw[i])), (tag, readd[i], lw[i])
        # (2) a single path cannot outweigh the sum over all paths
        assert lw[i] <= lz[i] + _LOGZ_ATOL, (tag, lw[i], lz[i])
        segs, pos = [], 0
        after = []
        for t in range(t_len):
            k = int(ep[i, t])
            if k >= 0:
                d = int(ed[i, t])
                # (3) tiling: every segment starts where the previous one ended
                assert int(es[i, t]) == pos, (tag, t, int(es[i, t]), pos)
                # (4) duration legality: d_min <= d <= D_k
                assert cfg.d_min <= d <= cfg.type_d_max(k), (tag, t, k, d)
                segs.append([k, pos, d])
                pos += d
            after.append(pos)
        assert int(ep[i, 0]) >= 0, (tag, "the first recognizer frame must emit")
        assert pos == s_len, (tag, "the segments do not cover the utterance", pos, s_len)
        # (5) the band: the unit position after every recognizer frame is within W of 3 (t + 1)
        for t, p in enumerate(after):
            assert abs(p - stride * (t + 1)) <= w, (tag, t, p)
        assert (ep[i, t_len:] < 0).all(), (tag, "an emit past the utterance's frames")
        row.update({"log_w": float(lw[i]), "log_z_tau1": float(lz[i]),
                    "log_w_minus_log_z": float(lw[i] - lz[i]),
                    "tokens": [PHONES[k] for k, _s, _d in segs], "segments": segs})
        if original_lens is not None:
            n_emit = sum(1 for k, _s, _d in segs if k != SIL_ID)
            row.update({"emitted_nonsil_tokens": n_emit,
                        "emitted_nonsil_rate_original_hz":
                            FRAME_RATE_HZ * n_emit / row["original_frames"]})
        rows.append(row)
    return {"rows": rows, "units": units, "dp": dp}


def emitted_rate_summary(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The A8 rate of a decode: pooled and utterance-mean emitted non-SIL rate over the decoded
    (non-impossible) rows, on the original-audio denominator (module conventions)."""
    live = [r for r in rows if not r["impossible"]]
    if not live or live[0].get("emitted_nonsil_tokens") is None:
        return None
    n_tok = sum(r["emitted_nonsil_tokens"] for r in live)
    orig = sum(r["original_frames"] for r in live)
    return {"n": len(live), "emitted_nonsil_tokens": n_tok, "original_frames": orig,
            "pooled_hz": FRAME_RATE_HZ * n_tok / orig,
            "utterance_mean_hz": sum(r["emitted_nonsil_rate_original_hz"] for r in live) / len(live),
            "band_hz": list(RATE_BAND_HZ),
            "definition": ("A8: 50 x (sum of non-SIL tokens in the tau = 1 generative posterior "
                           "decode) / (sum of original 50 Hz frames), pooled over the decoded "
                           "utterances; THE rate of A2's VOID, A3's probe PASS and NO ELIGIBLE "
                           "RESTART")}


# =================================================================================================
# 3. RETURNN forward steps and callbacks
# =================================================================================================
def _batch_inputs(model, extern_data):
    from ..model.train_step import seq_lens, seq_tags

    feats_ = extern_data[model.features_key]
    units_ = extern_data[model.units_key]
    units = units_.raw_tensor.long()
    lengths, unit_lens = seq_lens(feats_), seq_lens(units_)
    assert bool((lengths == unit_lens).all()), "features and units must share one clock"
    b = int(units.shape[0])
    tags = seq_tags(extern_data, model.seq_tag_key, b)
    # the train step's read of the original-audio length (the rate's denominator)
    original = extern_data["original_length"].raw_tensor.long().reshape(-1)
    assert int(original.shape[0]) == b and bool((original >= unit_lens.to(original.device)).all())
    return feats_, units, unit_lens, tags, original


def _mark_batch(fwd_ctx, feats_, batch_index: int, device):
    import torch

    import returnn.frontend as rf

    b_dim = feats_.dims[0]
    b = int(feats_.raw_tensor.shape[0])
    idx = torch.full((b,), batch_index, dtype=torch.int32, device=device)
    fwd_ctx.mark_as_output(rf.convert_to_tensor(idx, dims=[b_dim]), "genmarg_batch", dims=[b_dim])


def _settings(model, shuffle_seed) -> Dict[str, Any]:
    cfg = model.lattice_cfg
    return {
        "recognizer": "null (log_q = 0, float64)", "temperatures": list(TEMPERATURES),
        "prior_weight": float(model.prior_weight_at(1)), "anchor_weight": 0.0,
        "prior_history": model.prior_history.name, "prior_npz": getattr(model, "_genmarg_prior", None),
        "lattice_reduction": model.lattice_reduction, "dtype": "float64",
        "lattice": {"topology": cfg.topology, "recognizer_stride": cfg.recognizer_stride,
                    "band": cfg.band, "d_min": cfg.d_min, "d_max": cfg.d_max,
                    "d_max_sil": cfg.d_max_sil, "sil_id": cfg.sil_id},
        "shuffle_seed": shuffle_seed,
        "k2_term": "not computed (amendment A1: phi-independent under the null recognizer)",
        "tau2_banked": VALUE_LABELS["free_energy_tau2"],
        "rate": "gated (A8): the decode's emitted non-SIL rate, 50 x tokens / orig_length "
                "(gendecode.json); report only: E[N_nonSIL] under the tau = 1 generative "
                "posterior, Hz = 50 E / orig_length (original) and 50 E / S (retained)",
    }


def genmarg_forward_step(*, model, extern_data, shuffle_seed: Optional[int] = None, **kwargs):
    """Statistic (a) for one batch (module conventions)."""
    import returnn.frontend as rf

    fwd_ctx = rf.get_run_ctx()
    assert fwd_ctx.stage == "forward_step", fwd_ctx
    feats_, units, unit_lens, tags, original = _batch_inputs(model, extern_data)
    res = generative_marginals(model, units, unit_lens, tags, shuffle_seed=shuffle_seed)
    rows = marginal_rows(res, unit_lens.cpu().tolist(), tags, original.cpu().tolist())
    batch_index = len(model._genmarg_batches)
    model._genmarg_batches.append({"batch": batch_index, "tags": tags, "rows": rows,
                                   "settings": _settings(model, shuffle_seed)})
    _mark_batch(fwd_ctx, feats_, batch_index, units.device)


def gendecode_forward_step(*, model, extern_data, shuffle_seed: Optional[int] = None, **kwargs):
    """phi's generative posterior decode for one batch, validity-checked (module conventions)."""
    import returnn.frontend as rf

    fwd_ctx = rf.get_run_ctx()
    assert fwd_ctx.stage == "forward_step", fwd_ctx
    feats_, units, unit_lens, tags, original = _batch_inputs(model, extern_data)
    res = generative_decode(model, units, unit_lens, tags, shuffle_seed=shuffle_seed,
                            original_lens=original.cpu().tolist())
    batch_index = len(model._genmarg_batches)
    model._genmarg_batches.append({"batch": batch_index, "tags": tags, "rows": res["rows"],
                                   "settings": _settings(model, shuffle_seed)})
    _mark_batch(fwd_ctx, feats_, batch_index, units.device)


def _collect(model, seen, expected_utterances):
    batches = model._genmarg_batches
    for rec in batches:
        for tag in rec["tags"]:
            assert seen.get(tag) == rec["batch"], (tag, rec["batch"])
    rows = [r for rec in batches for r in rec["rows"]]
    assert len(rows) == len(seen) == len({r["tag"] for r in rows}), (len(rows), len(seen))
    if expected_utterances is not None:
        assert len(rows) == int(expected_utterances), (len(rows), expected_utterances)
    settings = batches[0]["settings"]
    assert all(rec["settings"] == settings for rec in batches)
    return rows, settings


def _callback_base():
    from returnn.forward_iface import ForwardCallbackIface

    class _Base(ForwardCallbackIface):
        def __init__(self):
            self._model = None
            self._seen: Dict[str, int] = {}

        def init(self, *, model, **kwargs):
            self._model = model
            self._seen = {}

        def process_seq(self, *, seq_tag, outputs, **kwargs):
            assert seq_tag not in self._seen, f"duplicate seq tag in forward: {seq_tag}"
            self._seen[str(seq_tag)] = int(outputs["genmarg_batch"].raw_tensor)

    return _Base


def GenMargCallback(*, name: str, dataset: str, expected_utterances: Optional[int] = None,
                    out_file: str = "genmarg.json", **_kwargs):
    """Forward callback factory: writes ``genmarg.json`` (statistic (a), module conventions)."""
    base = _callback_base()

    class _GenMargCallback(base):
        def finish(self, **kwargs):
            rows, settings = _collect(self._model, self._seen, expected_utterances)
            summary = summarise_marginals(rows)
            record = {"schema": "sae4a-lexlat-v2-genmarg-v1", "name": name, "dataset": dataset,
                      "load": self._model._genmarg_load, "settings": settings,
                      "conventions": __doc__.split("CONVENTIONS")[1].strip(),
                      "summary": summary,
                      "per_utterance": {r["tag"]: {k: v for k, v in r.items() if k != "tag"}
                                        for r in rows}}
            with open(out_file, "w") as fh:
                json.dump(record, fh, indent=1)
            print(record["conventions"], flush=True)
            print(f"genmarg {name} on {dataset}: {summary['n_utterances']} utterances, "
                  f"{summary['n_impossible']} impossible (never averaged); per-frame "
                  + "; ".join(f"{k} {summary[k]['utterance_mean_per_frame']:.6f} "
                              f"(pooled {summary[k]['pooled_per_frame']:.6f})"
                              for k in VALUE_KEYS if summary[k]), flush=True)
            r = summary["expected_nonsil_rate"]
            if r:
                print(f"genmarg {name} on {dataset}: REPORT ONLY (A8) expected non-SIL rate (tau = 1 "
                      "posterior) "
                      f"{r['original_hz']['pooled_hz']:.4f} Hz pooled / "
                      f"{r['original_hz']['utterance_mean_hz']:.4f} Hz utterance mean (original "
                      f"audio); {r['retained_hz']['pooled_hz']:.4f} Hz pooled (retained)", flush=True)

    return _GenMargCallback()


def GenDecodeCallback(*, name: str, dataset: str, expected_utterances: Optional[int] = None,
                      out_summary_file: str = "gendecode.json",
                      out_raw_file: str = "decode_raw.json", **_kwargs):
    """Forward callback factory: writes ``decode_raw.json`` and ``gendecode.json``."""
    base = _callback_base()

    class _GenDecodeCallback(base):
        def finish(self, **kwargs):
            rows, settings = _collect(self._model, self._seen, expected_utterances)
            live = [r for r in rows if not r["impossible"]]
            raw = {r["tag"]: r["tokens"] for r in live}
            gaps = [r["log_w_minus_log_z"] for r in live]
            record = {
                "schema": "sae4a-lexlat-v2-gendecode-v1", "name": name, "dataset": dataset,
                "load": self._model._genmarg_load, "settings": settings,
                "conventions": __doc__.split("CONVENTIONS")[1].strip(),
                "n_utterances": len(rows), "n_decoded": len(live),
                "n_impossible": len(rows) - len(live),
                "impossible_tags": sorted(r["tag"] for r in rows if r["impossible"]),
                "checks_passed": ["readd == max-plus max", "log_w <= log Z_1", "tiling",
                                  "d_min <= d <= D_k", "band after every recognizer frame"],
                "max_log_w_minus_log_z": max(gaps) if gaps else None,
                "mean_tokens": (sum(len(r["tokens"]) for r in live) / len(live)) if live else None,
                "emitted_nonsil_rate": emitted_rate_summary(rows),
                "per_utterance": {r["tag"]: {k: v for k, v in r.items() if k != "tag"} for r in rows},
            }
            with open(out_raw_file, "w") as fh:
                json.dump(raw, fh)
            with open(out_summary_file, "w") as fh:
                json.dump(record, fh, indent=1)
            print(f"gendecode {name} on {dataset}: {len(live)} / {len(rows)} decoded, "
                  f"{record['n_impossible']} impossible; every live path valid; "
                  f"max(log_w - log Z_1) = {record['max_log_w_minus_log_z']}", flush=True)
            r = record["emitted_nonsil_rate"]
            if r:
                print(f"gendecode {name} on {dataset}: EMITTED non-SIL rate (A8, the gated rate) "
                      f"{r['pooled_hz']:.4f} Hz pooled / {r['utterance_mean_hz']:.4f} Hz utterance "
                      f"mean over {r['n']} decoded utterances (band {RATE_BAND_HZ[0]}-"
                      f"{RATE_BAND_HZ[1]})", flush=True)

    return _GenDecodeCallback()


