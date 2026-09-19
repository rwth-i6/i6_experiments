"""``wav2vec_u_rev`` -- fairseq's wav2vec-U 2.0 generator plus the SAE reverse term (SAE §4a step 4).

The model is a thin subclass of the reference ``Wav2vec_U``: the GAN losses, the auxiliary MFCC CE,
the discriminator, the segmenter, the temperature schedule and the 1:1 G/D alternation are inherited
untouched.  The single delta is one extra entry in ``net_output["losses"]`` on GENERATOR steps:

    losses["rev"] = lam_rev * sample_size * mean_over_kept( -log Z_tau / S )

``sample_size`` (= the number of utterances in the batch) is the convention every other model-side
loss here follows, because fairseq's trainer divides the gradients by it; ``lam_rev`` is applied
inside the model rather than through ``criterion.loss_weights`` so that the reference criterion
config stays untouched.  The mean is over the sub-batch of ``rev_batch_utts`` utterances (0 = the
whole batch); the normalization does NOT depend on that count, so b is a cost/variance knob and not
a second weight.

phi (``emc.reverse.SegmentalReverseModel``) lives inside this model, so it is saved in and restored
from the fairseq checkpoint like any other parameter.  Its parameters are tagged
``param_group = "reverse"`` and are stepped by the composite optimizer's ``reverse`` group, which
``unpaired_audio_text_rev`` adds to the generator update.  A frozen-phi arm
(``rev_frozen_phi = <blank-free checkpoint>``) loads a donor phi and sets ``requires_grad = False``;
those parameters then never reach the optimizer, and the config must NOT declare a ``reverse`` group
(fairseq's composite optimizer asserts that its group names are exactly the tags it finds).

The term itself is in ``w2vu_rev.rev_term`` and calls the campaign's single lattice implementation.
"""

# NO `from __future__ import annotations` here: omegaconf 2.0.6 (fairseq 0.12.2's pin) reads the
# dataclass field types at registration and cannot resolve STRING annotations -- with the future
# import every field type becomes a string and hydra dies in `issubclass(type_, Enum)`.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from fairseq.models import register_model

# Registered by ``w2vu_rev/__init__.py``, which imports fairseq's own unsupervised user-dir first.
from unsupervised.models.wav2vec_u import Wav2vec_U, Wav2vec_UConfig

from w2vu_rev import rev_term as rt

logger = logging.getLogger(__name__)


@dataclass
class Wav2vec_U_RevConfig(Wav2vec_UConfig):
    lam_rev: float = field(default=0.0, metadata={"help": "weight of the reverse term"})
    rev_batch_utts: int = field(
        default=0, metadata={"help": "utterances per generator update the term is computed on (0 = all)"}
    )
    rev_tau: float = field(default=rt.REV_TAU, metadata={"help": "lattice temperature tau"})
    rev_float64: bool = field(default=rt.REV_FLOAT64, metadata={"help": "run the DP in fp64"})
    rev_phi_seed: int = field(default=0, metadata={"help": "cold initialization seed of phi"})
    rev_frozen_phi: Optional[str] = field(
        default=None, metadata={"help": "blank-free checkpoint to load phi from; phi is then frozen"}
    )


@register_model("wav2vec_u_rev", dataclass=Wav2vec_U_RevConfig)
class Wav2vec_U_Rev(Wav2vec_U):
    def __init__(self, cfg: Wav2vec_U_RevConfig, target_dict):
        super().__init__(cfg, target_dict)
        self.lam_rev = float(cfg.lam_rev)
        self.rev_batch_utts = int(cfg.rev_batch_utts)
        self.rev_cfg = rt.RevTermConfig(tau=float(cfg.rev_tau), float64=bool(cfg.rev_float64))
        assert self.lam_rev > 0, (
            "wav2vec_u_rev is the reverse-term model; at lam_rev = 0 the reproduction's own "
            "wav2vec_u must be used instead, so that the weight-0 code path is byte-identical"
        )
        # The term reads the generator's stride-3 logits BEFORE logit_segment; that is only the
        # recognizer's own output rate when pre_segment is the identity.
        assert cfg.segmentation.type.name in ("NONE", "JOIN"), (
            f"segmentation {cfg.segmentation.type} changes the time axis before the generator; the "
            "reverse term's T = ceil(S/3) alignment assumes NONE or JOIN"
        )

        self.phi = rt.build_phi(seed=int(cfg.rev_phi_seed))
        self.rev_frozen = cfg.rev_frozen_phi is not None
        if self.rev_frozen:
            rt.load_phi_state(cfg.rev_frozen_phi, self.phi)
            for p in self.phi.parameters():
                p.requires_grad = False
            logger.info(f"reverse model frozen, loaded from {cfg.rev_frozen_phi}")
        else:
            for p in self.phi.parameters():
                p.param_group = "reverse"
            logger.info(f"reverse model trained in the 'reverse' optimizer group, seed {cfg.rev_phi_seed}")

        cols = rt.phone_columns(target_dict.index, target_dict.unk())
        self.register_buffer("phone_columns", torch.tensor(cols, dtype=torch.long), persistent=False)

        # The generator's pre-segmentation output is not returned by Wav2vec_U.forward, so it is
        # captured with a standard forward hook instead of copying the reference forward.
        self._gen_result = None
        self.generator.register_forward_hook(self._capture_generator)

        # the sub-batch draw (see reverse_loss); CPU generator so it is device-independent
        self._rev_rng = torch.Generator()
        self._rev_rng.manual_seed(int(cfg.rev_phi_seed))

    def _capture_generator(self, module, inputs, output):
        self._gen_result = output

    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        dense_x_only=False,
        segment=True,
        aux_target=None,
        rev_units=None,
        rev_unit_lens=None,
        rev_eta=None,
    ):
        result = super().forward(
            features,
            padding_mask,
            random_label=random_label,
            dense_x_only=dense_x_only,
            segment=segment,
            aux_target=aux_target,
        )
        if dense_x_only or self.discriminator is None:
            return result
        # Generator steps only.  Validation keeps the reproduction's path exactly (the term is a
        # training loss; the dev reverse likelihood is read by its own job).
        if result["d_steps"] or not self.training:
            return result
        assert rev_units is not None and rev_eta is not None and rev_unit_lens is not None, (
            "the reverse term needs rev_units / rev_unit_lens / rev_eta in net_input; the dataset "
            "wrapper of unpaired_audio_text_rev supplies them"
        )
        gen, self._gen_result = self._gen_result, None  # never hold the graph past this step
        loss, stats = self.reverse_loss(gen, rev_units, rev_unit_lens, rev_eta)
        result["losses"]["rev"] = loss
        result.setdefault("logs", {}).update(stats)
        return result

    def reverse_loss(self, gen, rev_units, rev_unit_lens, rev_eta):
        """``(lam_rev * sample_size * mean(-log Z_tau / S), stats)`` for the captured generator output."""
        assert gen is not None, "the generator forward hook did not fire"
        dense_x, dense_padding_mask = gen["dense_x"], gen["dense_padding_mask"]
        sample_size = dense_x.size(0)

        feat_lens = (~dense_padding_mask).sum(-1).long()
        unit_lens = rev_unit_lens.long()
        expected = torch.div(unit_lens + 2, 3, rounding_mode="floor")  # ceil(S / 3), stride 3
        assert torch.equal(feat_lens, expected), (
            "generator output length != ceil(S/3): the unit stream and the feature stream are not "
            "on the same retained 50 Hz clock"
        )

        # The sub-batch: b utterances drawn UNIFORMLY at random per generator update (a random
        # subset, not the first b rows, because fairseq's batch sampler orders a batch by length).
        # The draw comes from a private generator seeded by rev_phi_seed, so a run is reproducible
        # from its start; the normalization below does not depend on b, so b is a cost knob and the
        # term stays an unbiased estimate of the full-batch one.
        b = self.rev_batch_utts if 0 < self.rev_batch_utts < sample_size else sample_size
        if b < sample_size:
            sel = torch.randperm(sample_size, generator=self._rev_rng)[:b].to(dense_x.device)
        else:
            sel = torch.arange(sample_size, device=dense_x.device)
        unit_lens = unit_lens.index_select(0, sel)
        feat_lens = feat_lens.index_select(0, sel)
        t_sel, s_sel = int(feat_lens.max()), int(unit_lens.max())
        log_q = rt.log_q_from_dense(dense_x.index_select(0, sel)[:, :t_sel], self.phone_columns)
        out = rt.reverse_term(
            log_q=log_q,
            units=rev_units.index_select(0, sel)[:, :s_sel].long(),
            unit_lens=unit_lens,
            feat_lens=feat_lens,
            eta=rev_eta.index_select(0, sel).to(log_q.dtype),
            phi=self.phi,
            cfg=self.rev_cfg,
        )
        stats = dict(out.stats)
        stats["rev_batch_utts"] = float(b)
        return self.lam_rev * sample_size * out.loss, stats
