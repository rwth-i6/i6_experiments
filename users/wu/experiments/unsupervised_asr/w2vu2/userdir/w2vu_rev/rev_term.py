"""SAE §4a step 4 -- the project's alignment-sum reverse term L_rev, ported onto GAN logits.

This module is deliberately **fairseq-free**: it only knows torch and ``speech_llm.sae.emc``, so the
equality test can call it side by side with a direct ``build_segment_table`` + ``lattice_loss`` call.
There is exactly ONE implementation of the lattice in this campaign -- the one in
``recipe/2025-10-speech-llm/src/speech_llm/sae/emc/lattice.py``, which the fairseq worker imports
through the PYTHONPATH that ``settings.py`` already hands every ``requires_env = "w2vu"`` job
(``recipe/2025-10-speech-llm/src`` is on it).  Nothing here re-derives the DP, the segment table or
the duration head.

The term (SAE_4A_attrib.md step 4, as amended 2026-09-19, and SAE_4A_blankfree.md:65-146):

    L_rev(b) = -log Z_tau(b) / S_b ,   tau = 1  (plain marginal likelihood),
               beta = 0              (no trigram factor inside the objective),
               blank-free topology, d_min = 2, D = 25, D_sil = 50, band |s - 3t| <= 25,
               T_b = ceil(S_b / 3)   (the generator's stride-3 output rate),
               z = the enc50 K = 500 unit stream on the retained (VAD-masked) 50 Hz clock,
               phi = ``emc.reverse.SegmentalReverseModel`` with the frozen per-utterance eta.

``beta = 0`` still needs a prior table of the right shape for the DP, so an all-zero ``[|h|, K]``
table is passed and multiplied by ``prior_weight = 0``; the history axis is the BIGRAM one
(``bigram_history``), i.e. the lattice state keeps only the last phone, which is what the blank-free
repeat rule needs and nothing more.

The batch mean is taken over the utterances whose ``Z_tau > 0`` (``z_zero`` is reported, never
averaged in) -- the blank-free train step's own convention
(``speech_llm/prefix_lm/model/train_steps/sae_blankfree.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

import torch

from speech_llm.sae.emc import prior as _prior
from speech_llm.sae.emc.lattice import LatticeConfig, bigram_history, build_segment_table, lattice_loss
from speech_llm.sae.emc.reverse import ReverseConfig, SegmentalReverseModel

# The topology of SAE_4A_blankfree.md:65-146, verbatim: 39 ARPAbet + SIL, SIL last, W = 25,
# d_min = 2, D = 25, D_sil = 50, the 50 Hz reverse clock and the recognizer's stride 3.
REV_LATTICE_CFG = LatticeConfig(
    n_phones=_prior.N_TYPES,
    sil_id=_prior.SIL_ID,
    band=25,
    d_min=2,
    d_max=25,
    d_max_sil=50,
    frame_rate_hz=50.0,
    topology="blankfree",
    recognizer_stride=3,
)

# phi's shape, as registered for the blank-free bed (emc.reverse.ReverseConfig defaults; the unit
# vocabulary is the enc50 K = 500 codebook and eta is the frozen PCA-16 speaker vector).
REV_REVERSE_CFG = ReverseConfig()

# Inherited from the registered blank-free run (blankfree_train_jobs.build_blankfree_train_config):
# fp64 DP, the frame-stride-32 backward recomputation.  ``reduction = "auto"`` is the module's own
# choice per history and keeps the bit-identical elementwise path at the bigram history used here.
REV_FLOAT64 = True
REV_CHECKPOINT = 32
REV_REDUCTION = "auto"
# tau = 1: the design review's amendment of 2026-09-19 (SAE_4A_attrib.md "Design-review amendments").
REV_TAU = 1.0
# beta = 0: no trigram factor inside the ported term (SAE_4A_attrib.md step 4).
REV_PRIOR_WEIGHT = 0.0


@dataclass(frozen=True)
class RevTermConfig:
    """The knobs of one ported term.  Everything not listed is fixed by the constants above."""

    tau: float = REV_TAU
    prior_weight: float = REV_PRIOR_WEIGHT
    float64: bool = REV_FLOAT64
    reduction: str = REV_REDUCTION
    checkpoint: int = REV_CHECKPOINT


class RevTermOutput(NamedTuple):
    loss: torch.Tensor          # scalar: mean over the kept utterances of -log Z_tau / S
    stats: Dict[str, float]     # per-batch diagnostics (floats, safe for a logging_output)
    n_kept: int                 # utterances with Z_tau > 0 that entered the mean
    per_utt: torch.Tensor       # [b] detached -log Z_tau / S, per utterance (0 where z_zero)
    keep: torch.Tensor          # [b] bool: Z_tau > 0


def build_phi(*, seed: Optional[int] = None, cfg: ReverseConfig = REV_REVERSE_CFG) -> SegmentalReverseModel:
    """A fresh reverse model.  ``seed`` reproduces ``reset_parameters``' cold initialization."""
    phi = SegmentalReverseModel(cfg)
    if seed is not None:
        phi.reset_parameters(seed)
    return phi


def load_phi_state(path: str, phi: SegmentalReverseModel) -> None:
    """Load the reverse-model parameters of a blank-free RETURNN checkpoint into ``phi``.

    The checkpoint holds the whole SAE model (``{"model": {"recognizer.*", "reverse.*", ...}}``); only
    the ``reverse.`` sub-tree is taken, and it must be complete -- a partial load would silently mix a
    donor phi with a cold one.
    """
    state = torch.load(path, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    sub = {k[len("reverse."):]: v for k, v in sd.items() if k.startswith("reverse.")}
    assert sub, f"no 'reverse.*' parameters in {path}"
    res = phi.load_state_dict(sub, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res


def phone_columns(index_fn, unk_index: int) -> List[int]:
    """Map the 40 lattice phone ids to their columns in the generator's output vocabulary.

    ``index_fn`` is fairseq's ``Dictionary.index``.  SIL is spelled ``<SIL>`` there and ``SIL`` in ``emc.prior.PHONES``; every
    other symbol is identical.  A phone that maps to ``unk`` is an error (the two inventories must be
    the same 40 types, or the term would be scoring the wrong columns).
    """
    cols = []
    for p in _prior.PHONES:
        sym = "<SIL>" if p == _prior.SIL else p
        idx = int(index_fn(sym))
        assert idx != unk_index, f"phone {sym!r} is not in the generator dictionary"
        cols.append(idx)
    assert len(set(cols)) == len(cols) == _prior.N_TYPES
    return cols


def log_q_from_dense(dense_x: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """``[B, T, 44]`` generator logits -> ``[B, T, 40]`` log-probabilities in lattice phone order.

    The 4 fairseq specials (bos/pad/eos/unk) are dropped and the 40 phone columns are RE-NORMALIZED,
    because the lattice's emission factor is a distribution over the 40 phone types (the blank-free
    recognizer emits exactly those 40 and nothing else).  The GAN's own softmax over all 44 columns is
    untouched -- this is a second, term-local read of the same logits.
    """
    return torch.log_softmax(dense_x.index_select(-1, columns), dim=-1)


def reverse_term(
    *,
    log_q: torch.Tensor,        # [b, T, 40] log-probabilities (lattice phone order)
    units: torch.Tensor,        # [b, S] long, the enc50 K = 500 stream on the retained clock
    unit_lens: torch.Tensor,    # [b] long, S_b
    feat_lens: torch.Tensor,    # [b] long, T_b = ceil(S_b / 3)
    eta: torch.Tensor,          # [b, 16] frozen speaker vectors
    phi: SegmentalReverseModel,
    cfg: RevTermConfig = RevTermConfig(),
    lattice_cfg: LatticeConfig = REV_LATTICE_CFG,
) -> RevTermOutput:
    """The ported term for one (sub)batch: mean over kept utterances of ``-log Z_tau / S``.

    Differentiable in ``log_q`` (theta) and in phi through the segment table, by the same arc-posterior
    surrogate the blank-free train step uses -- this function only assembles the call.
    """
    assert log_q.dim() == 3 and log_q.shape[-1] == lattice_cfg.n_phones, log_q.shape
    assert units.dim() == 2 and units.shape[0] == log_q.shape[0], units.shape
    assert torch.all(unit_lens >= 1) and torch.all(feat_lens >= 1)
    assert units.shape[1] >= int(unit_lens.max()), (units.shape, int(unit_lens.max()))
    assert log_q.shape[1] >= int(feat_lens.max()), (log_q.shape, int(feat_lens.max()))

    hist = bigram_history(lattice_cfg)
    seg = build_segment_table(phi, units, eta)
    dp_log_q, dp_seg = log_q, seg
    if cfg.float64:
        dp_log_q, dp_seg = log_q.double(), seg.double()
    # beta = 0: the table is never read for its values, only for its shape.
    prior_log_bi = dp_log_q.new_zeros(hist.n_hist, lattice_cfg.n_phones)

    loss_utt, out = lattice_loss(
        dp_log_q,
        dp_seg,
        prior_log_bi=prior_log_bi,
        feat_lens=feat_lens,
        unit_lens=unit_lens,
        cfg=lattice_cfg,
        temperature=cfg.tau,
        anchor_weight=0.0,
        prior_weight=cfg.prior_weight,
        log_q_init=None,
        collect_stats=False,          # expected_prior is the beta term; it is off here
        # The GAN batches 160 utterances (dataset.batch_size); the module's MAX_UTTS_PER_BATCH = 128
        # backstop was fitted for the RETURNN bed at stride 1, so it is downgraded to a warning here.
        enforce_budget=False,
        history=hist,
        reduction=cfg.reduction,
        checkpoint=cfg.checkpoint,
    )
    keep = (~out.z_zero).to(loss_utt.dtype)
    n_keep = keep.sum().clamp(min=1)
    retained = unit_lens.to(loss_utt.dtype)
    loss = ((loss_utt / retained) * keep).sum() / n_keep
    with torch.no_grad():
        stats = {
            # the loss itself: -log Z_tau per retained frame (lower is better)
            "rev_per_frame": float(loss.detach()),
            # the same quantity as a LOG-LIKELIHOOD (higher is better); this is the activity readout
            "rev_logz_per_frame": float(-loss.detach()),
            # E[sum_u G] per retained frame under the arc posterior (the reverse score alone)
            "rev_expected_per_frame": float(((out.expected_reverse / retained) * keep).sum() / n_keep),
            "rev_expected_tokens": float((out.expected_tokens * keep).sum() / n_keep),
            "rev_z_zero_frac": float(out.z_zero.to(loss_utt.dtype).mean()),
            "rev_utts": float(keep.sum()),
        }
    return RevTermOutput(
        loss=loss.to(log_q.dtype),
        stats=stats,
        n_kept=int(keep.sum()),
        per_utt=(loss_utt / retained).detach() * keep,
        keep=~out.z_zero,
    )
