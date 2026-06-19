"""End-to-end forward/backward smoke test for two_level_v1.Model (run with conda speech_llm python).

Validates the integration the Sisyphus run can't cheaply check up front:
  * the frozen lower stack -> CIF -> high encoder -> head wires end-to-end on real shapes;
  * variable K / span mask / CE plumbing produces finite losses;
  * gradient flows into the TRAINABLE parts (cif_alpha, high_encoder, mask_emb, pred_head) and NOT into
    the FROZEN lower stack (encoder / feature_extraction have requires_grad=False).
Run:
  PYTHONPATH=recipe:recipe/i6_models:recipe/returnn \
    <conda_py> recipe/i6_experiments/users/wu/experiments/ssl/pytorch_networks/two_level/test_model.py
"""

from dataclasses import asdict

import torch

from i6_experiments.users.wu.experiments.ssl.pytorch_networks.two_level.two_level_v1 import Model
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.two_level.two_level_v1_cfg import TwoLevelConfig
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.common.conformer import (
    default_encoder_config, default_high_encoder_config,
)
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.two_level.parts.cif import quantity_loss
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.ls_logmel_stats import LOGMEL_MEAN, LOGMEL_STD


def build_cfg():
    return TwoLevelConfig(
        feature_extraction_config=LogMelFeatureExtractionV1Config(
            sample_rate=16000, win_size=0.025, hop_size=0.01, f_min=60, f_max=7600,
            min_amp=1e-10, num_filters=80, center=False,
        ),
        encoder_config=default_encoder_config(num_layers=9),
        high_encoder_config=default_high_encoder_config(num_layers=9, rel_pos_clip=96),
        global_mean=list(LOGMEL_MEAN), global_std=list(LOGMEL_STD),
        lower_layer_index=8,
        cif_alpha_kernel_size=5, target_rate_hz=12.5, frame_rate_hz=25.0, lambda_qty=1.0,
        num_clusters=128, mask_prob=0.2, mask_length=3, min_masks=1,  # higher mask_prob -> ensure masked>0 in test
    )


def main():
    torch.manual_seed(0)
    cfg = build_cfg()
    model = Model(asdict(cfg), codebook=None)
    model.train()

    b = 2
    lengths = torch.tensor([24000, 18000])  # ~1.5 s, ~1.1 s @ 16 kHz
    raw = torch.randn(b, int(lengths.max()))

    logits, targets, loss_mask, z_mask, diag, frame_len = model(raw_audio=raw, raw_audio_len=lengths)
    print("frame_len (25Hz):", frame_len.tolist())
    print("K (fires):", diag["fires"].tolist(), " z_mask valid:", z_mask.sum(1).tolist())
    print("logits:", tuple(logits.shape), " targets:", tuple(targets.shape),
          " masked tokens:", int(loss_mask.sum()))
    assert logits.shape[:2] == targets.shape == z_mask.shape
    assert int(loss_mask.sum()) > 0, "no masked tokens -> increase mask_prob/length for the test"

    ce = torch.nn.functional.cross_entropy(logits[loss_mask], targets[loss_mask])
    qty = quantity_loss(diag["sum_alpha"], frame_len, target_rate_hz=cfg.target_rate_hz, frame_rate_hz=25.0)
    loss = ce + cfg.lambda_qty * qty
    assert torch.isfinite(loss), loss
    print(f"ce={float(ce):.4f}  qty={float(qty):.4f}")
    loss.backward()

    # gradient routing
    def gnorm(mod):
        g = [p.grad.abs().sum() for p in mod.parameters() if p.grad is not None]
        return float(sum(g)) if g else 0.0

    assert gnorm(model.cif_alpha) > 0, "no grad into CIF predictor!"
    assert gnorm(model.high_encoder) > 0, "no grad into high encoder!"
    assert gnorm(model.pred_head) > 0, "no grad into prediction head!"
    assert model.mask_emb.grad is not None and model.mask_emb.grad.abs().sum() > 0
    # frozen lower stack: requires_grad False AND no grad accumulated
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.feature_extraction.parameters())
    assert gnorm(model.encoder) == 0.0, "gradient leaked into the frozen lower encoder!"
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M   frozen params: {n_frozen/1e6:.1f}M")
    print(f"cif|high|head grad norms: {gnorm(model.cif_alpha):.3f} | "
          f"{gnorm(model.high_encoder):.1f} | {gnorm(model.pred_head):.3f}")
    print("\nMODEL FORWARD/BACKWARD SMOKE PASSED")


if __name__ == "__main__":
    main()
