"""
Structural smoke test for the BEST-RQ model's forward + loss + backward (no RETURNN run_ctx).

Uses a deliberately small encoder (fast on CPU) but the real BEST-RQ structure. Verifies:
* enc_out / targets / loss_mask time dims align and loss_mask has masked positions,
* the averaged-CE loss is finite and backprops,
* gradients reach the encoder + heads but NOT the frozen quantizer (no params there).

Run:
    PYTHONPATH=recipe:recipe/i6_models <conda-python> \
      recipe/i6_experiments/users/wu/experiments/ssl/pytorch_networks/best_rq/test_model.py
"""

from __future__ import annotations

import torch

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.best_rq_v1 import Model
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.common.conformer import (
    ConformerEncoderConfig,
    ConformerPosEmbConfig,
    VGGFrontendConfig,
)
from dataclasses import asdict


def _small_config():
    feat = LogMelFeatureExtractionV1Config(
        sample_rate=16000, win_size=0.025, hop_size=0.01, f_min=60, f_max=7600,
        min_amp=1e-10, num_filters=80, center=False,
    )
    frontend = VGGFrontendConfig(
        in_features=80, conv1_channels=32, conv2_channels=32, conv3_channels=64, conv4_channels=64,
        conv_kernel_size=(3, 3), conv_padding=None,
        pool1_kernel_size=(2, 1), pool1_stride=(2, 1), pool1_padding=None,
        pool2_kernel_size=(2, 1), pool2_stride=(2, 1), pool2_padding=None,
        out_features=128, activation_str="ReLU",
    )
    enc = ConformerEncoderConfig(
        frontend_config=frontend,
        pos_emb_config=ConformerPosEmbConfig(
            learnable_pos_emb=False, rel_pos_clip=None, with_linear_pos=True,
            with_pos_bias=True, separate_pos_emb_per_head=True, pos_emb_dropout=0.0,
        ),
        conformer_size=128, num_layers=2, num_heads=4, ff_dim=256,
        att_weights_dropout=0.1, conv_dropout=0.1, ff_dropout=0.1, mhsa_dropout=0.1,
        mhsa_with_bias=True, conv_kernel_size=31, dropout_broadcast_axes=None,
        module_list=["ff", "conv", "mhsa", "ff"], module_scales=[0.5, 1.0, 1.0, 0.5],
    )
    from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.ls_logmel_stats import (
        LOGMEL_MEAN, LOGMEL_STD,
    )
    return {
        "feature_extraction_config": asdict(feat),
        "encoder_config": _enc_to_dict(enc),
        "global_mean": LOGMEL_MEAN, "global_std": LOGMEL_STD,
        "stack_size": 4, "codebook_dim": 16, "vocab_size": 8192, "num_codebooks": 4,
        "quantizer_seed": 42, "mask_prob": 0.04, "mask_length": 10, "min_masks": 1, "noise_std": 0.1,
    }


def _enc_to_dict(enc: ConformerEncoderConfig):
    d = asdict(enc)  # asdict recurses dataclasses incl. the frontend (keeps activation_str)
    return d


def main():
    torch.manual_seed(0)
    model = Model(_small_config())
    model.train()

    b = 3
    lens = torch.tensor([16000 * 8, 16000 * 5, 16000 * 2])  # 8s/5s/2s
    maxlen = int(lens.max())
    raw = torch.randn(b, maxlen)

    enc_out, targets, loss_mask = model(raw_audio=raw, raw_audio_len=lens)
    print("enc_out", tuple(enc_out.shape), "targets", tuple(targets.shape), "loss_mask", tuple(loss_mask.shape))
    assert enc_out.shape[0] == b and targets.shape[0] == b
    assert enc_out.shape[1] == targets.shape[1] == loss_mask.shape[1]
    assert targets.shape[2] == 4
    nm = int(loss_mask.sum())
    print("num masked positions:", nm, "of", int((torch.arange(loss_mask.shape[1])[None] < (lens // 4 // 4)[:, None]).sum()))
    assert nm > 0

    masked_enc = enc_out[loss_mask]
    masked_tgt = targets[loss_mask]
    loss = masked_enc.new_zeros(())
    for n in range(model.num_codebooks):
        loss = loss + torch.nn.functional.cross_entropy(model.heads[n](masked_enc), masked_tgt[:, n])
    loss = loss / model.num_codebooks
    print("loss", float(loss))
    assert torch.isfinite(loss)
    loss.backward()

    # grads reach encoder + heads
    enc_grad = sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    head_grad = sum(p.grad.abs().sum() for p in model.heads.parameters() if p.grad is not None)
    assert float(enc_grad) > 0 and float(head_grad) > 0
    # quantizer has no parameters at all (frozen buffers)
    assert len(list(model.quantizer.parameters())) == 0
    print("ok: loss finite, grads flow to encoder+heads, quantizer frozen (no params)")
    print("ALL MODEL TESTS PASSED")


if __name__ == "__main__":
    main()
