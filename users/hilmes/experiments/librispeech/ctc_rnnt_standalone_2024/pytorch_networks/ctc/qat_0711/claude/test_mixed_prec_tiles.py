"""Manual verification script for per-tile (128x128) mixed-precision weight quantization.

Keep this script; it is meant to be re-run manually after changes. Run from the experiment
root (/u/hilmes/experiments/asr_2023):

    PYTHONPATH=recipe:i6_models_full:/u/hilmes/src/MiniReturnn \
        python3 recipe/i6_experiments/users/hilmes/experiments/librispeech/\
ctc_rnnt_standalone_2024/pytorch_networks/ctc/qat_0711/claude/test_mixed_prec_tiles.py

The script also inserts the paths itself, so a plain `python3 <script>` works too.

Note on tolerances: even with ideal_programming the memristor path has residual error from
DAC/ADC conversion, device non-linearity and sampled read noise, so memristor comparisons are
tolerance/correlation based, never exact. Bit-exact checks are only used where the same
memristor state is evaluated with the same RNG seed.
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve()
_exp_root = _here.parents[11]  # .../experiments/asr_2023
for p in [_exp_root / "recipe", _exp_root / "i6_models_full", Path("/u/hilmes/src/MiniReturnn")]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
from dataclasses import asdict

from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear

_pkg = "i6_experiments.users.hilmes.experiments.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.qat_0711.claude"
import importlib

cfg_mod = importlib.import_module(_pkg + ".memristor_v16_dynmic_prec_cfg")
modules = importlib.import_module(_pkg + ".memristor_v16_dynmic_prec_modules")
net = importlib.import_module(_pkg + ".memristor_v16_dynmic_prec")
net_mem_inited = importlib.import_module(_pkg + ".memristor_v16_dynmic_prec_mem_inited")

SubMatrixPrecision = cfg_mod.SubMatrixPrecision
split_ff_spec = cfg_mod.split_ff_spec
split_mhsa_spec = cfg_mod.split_mhsa_spec
split_conv_spec = cfg_mod.split_conv_spec
GaussianWeightNoiseConfig = cfg_mod.GaussianWeightNoiseConfig
BitFlipWeightNoiseConfig = cfg_mod.BitFlipWeightNoiseConfig

LinearQuant = modules.LinearQuant
ActivationQuantizer = modules.ActivationQuantizer
WeightQuantizer = modules.WeightQuantizer
TiledWeightQuantizer = modules.TiledWeightQuantizer
MixedPrecisionTiledMemristorLinear = modules.MixedPrecisionTiledMemristorLinear
convert_linear_to_memristor = modules.convert_linear_to_memristor
make_inference_memristor_linear = modules.make_inference_memristor_linear

HW_SETTINGS = DacAdcHardwareSettings(
    input_bits=8,
    output_precision_bits=4,
    output_range_bits=4,
    hardware_input_vmax=0.6,
    hardware_output_current_scaling=8020.0,
)
IDEAL_CORRECTION = CycleCorrectionSettings(
    num_cycles=None, test_input_value=None, relative_deviation=None, ideal_programming=True
)


def _make_linear(out_features, in_features, spec, seed=42):
    torch.manual_seed(seed)
    return LinearQuant(
        in_features=in_features,
        out_features=out_features,
        weight_bit_prec=spec,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        bias=True,
    )


def _make_act_quant():
    return ActivationQuantizer(
        bit_precision=8,
        dtype=torch.qint8,
        method="per_tensor_symmetric",
        channel_axis=1,
        moving_avrg=None,
    )


def _calibrated_pair(spec, out_features=256, in_features=256, seed=42):
    """LinearQuant + ActivationQuantizer with observers calibrated on a fixed input."""
    torch.manual_seed(seed)
    x = torch.randn(8, in_features) * 0.5
    lq = _make_linear(out_features, in_features, spec, seed=seed)
    aq = _make_act_quant()
    lq.train(), aq.train()
    _ = lq(aq(x))
    lq.eval(), aq.eval()
    return lq, aq, x


def test_spec_normalization():
    assert SubMatrixPrecision.from_spec(8, 256, 256, name="t") is None
    assert SubMatrixPrecision.from_spec(1.5, 256, 256, name="t") is None

    smp = SubMatrixPrecision.from_spec([8, 6], 256, 256, name="t")
    assert smp.grid == [[8, 8], [6, 6]], smp.grid  # 1D shorthand: per output tile row

    smp = SubMatrixPrecision.from_spec([[8, 6], [4, 8]], 256, 256, name="t")
    assert smp.grid == [[8, 6], [4, 8]]
    tiles = list(smp.iter_tiles())
    assert tiles[1][:4] == (0, 1, slice(0, 128), slice(128, 256))
    assert smp.max_precision == 8

    # edge tiles are clipped (e.g. 200 in_features -> second column tile is 72 wide)
    smp = SubMatrixPrecision.from_spec([[8, 6]], 100, 200, name="t")
    assert list(smp.iter_tiles())[1][3] == slice(128, 200)

    for bad in [[[8, 6]], [8, 6, 4], [[8, 6], [4]], [[8, 5.5], [4, 8]]]:
        try:
            SubMatrixPrecision.from_spec(bad, 256, 256, name="badcase")
        except AssertionError as e:
            assert "badcase" in str(e) or "precision" in str(e)
        else:
            raise AssertionError(f"spec {bad} should have been rejected")

    assert split_ff_spec(8) == (8, 8)
    assert split_ff_spec({"lin_1": [[8, 6]], "lin_2": 4}) == ([[8, 6]], 4)
    assert split_mhsa_spec(6, with_linear_pos=False) == (6, 6, 6)
    assert split_mhsa_spec({"W_i": [8], "W_o": 4, "learn_emb": 8}, with_linear_pos=True) == ([8], 4, 8)
    assert split_conv_spec({"pconv_1": [[8]], "pconv_2": 6, "dconv": 4}) == ([[8]], 6, 4)
    for fn, bad in [
        (split_ff_spec, [8, 6]),  # bare list at module level
        (split_ff_spec, {"lin_1": 8}),  # missing key
        (lambda s: split_mhsa_spec(s, with_linear_pos=False), {"W_i": 8, "W_o": 8, "learn_emb": 8}),
        (split_conv_spec, {"pconv_1": 8, "pconv_2": 8, "dconv": [[8]]}),  # dconv non-scalar
    ]:
        try:
            fn(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{bad} should have been rejected")
    print("PASS test_spec_normalization")


def test_qat_forward_backward():
    lq, aq, x = _calibrated_pair([[8, 6], [4, 8]])
    assert isinstance(lq.weight_quantizer, TiledWeightQuantizer)
    lq.train()
    out = lq(x)
    out.square().sum().backward()
    g = lq.weight.grad
    for r_sl, c_sl in [(slice(0, 128),) * 2, (slice(0, 128), slice(128, 256)),
                       (slice(128, 256), slice(0, 128)), (slice(128, 256),) * 2]:
        tile_grad = g[r_sl, c_sl]
        assert torch.isfinite(tile_grad).all() and tile_grad.abs().sum() > 0
    # per-tile observers hold per-tile scales
    scales = [lq.weight_quantizer.get_tile_quantizer(r, c).scale.item() for r in range(2) for c in range(2)]
    assert all(s > 0 for s in scales)
    print(f"PASS test_qat_forward_backward (tile scales: {[f'{s:.5f}' for s in scales]})")


def test_scalar_equivalence():
    """A [[8]] grid on a 128x128 matrix must behave exactly like scalar precision 8."""
    lq_scalar = _make_linear(128, 128, 8)
    lq_grid = _make_linear(128, 128, [[8]])
    with torch.no_grad():
        lq_grid.weight.copy_(lq_scalar.weight)
        lq_grid.bias.copy_(lq_scalar.bias)
    x = torch.randn(4, 128)
    lq_scalar.train(), lq_grid.train()
    out_s, out_g = lq_scalar(x), lq_grid(x)
    assert torch.equal(out_s, out_g), (out_s - out_g).abs().max()
    lq_scalar.eval(), lq_grid.eval()
    assert torch.equal(lq_scalar(x), lq_grid(x))
    print("PASS test_scalar_equivalence")


def test_legacy_state_dict_layout():
    """Scalar specs must keep the exact legacy module tree / state-dict keys."""
    lq = _make_linear(256, 256, 8)
    keys = set(lq.state_dict().keys())
    assert isinstance(lq.weight_quantizer, WeightQuantizer)
    assert any(k.startswith("weight_quantizer.observer.") for k in keys), keys
    assert not any("quantizers" in k for k in keys), keys
    print("PASS test_legacy_state_dict_layout")


def test_uniform_checkpoint_into_tiled():
    """A uniform-precision checkpoint has a single weight observer; a mixed model expects one
    observer per tile. There is no broadcast shim -- loading a uniform checkpoint into a tiled
    model is handled at the RETURNN level (preload `allowed_missing_suffix` for the observer
    buffers, which then recalibrate during finetuning). This test pins that contract: the tile
    observers are reported missing and the uniform observer is unexpected."""
    lq_scalar, _, _ = _calibrated_pair(8)
    sd = lq_scalar.state_dict()
    lq_grid = _make_linear(256, 256, [[8, 6], [4, 8]], seed=7)
    missing, unexpected = lq_grid.load_state_dict(sd, strict=False)
    missing_obs = [k for k in missing if "observer" in k]
    assert missing_obs and all("weight_quantizer.quantizers." in k for k in missing_obs), missing
    assert any(k.startswith("weight_quantizer.observer.") for k in unexpected), unexpected
    # weight/bias still load fine; only the observers differ in layout
    assert torch.equal(lq_grid.weight, lq_scalar.weight)
    print("PASS test_uniform_checkpoint_into_tiled")


def test_noise_per_tile():
    lq, _, _ = _calibrated_pair([[8, 6], [4, 8]])
    lq.weight_quantizer.set_scale_and_zp()
    w = lq.weight.data
    for noise in [GaussianWeightNoiseConfig(dev=0.05, start_epoch=0),
                  BitFlipWeightNoiseConfig(p=0.05, start_epoch=0)]:
        torch.manual_seed(0)
        noisy = noise.apply(w, lq.weight_quantizer, 0, training=False)
        assert noisy.shape == w.shape
        assert not torch.equal(noisy, w), type(noise).__name__
    print("PASS test_noise_per_tile")


def _convert(lq, aq):
    return convert_linear_to_memristor(
        linear_quant=lq,
        activation_quant=aq,
        converter_hardware_settings=HW_SETTINGS,
        num_cycles=0,
        correction_settings=IDEAL_CORRECTION,
    )


def _seeded_forward(module, x, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    with torch.no_grad():
        return module(x)


def test_conversion_numerics():
    """Mixed wrapper vs uniform TiledMemristorLinear, both vs the float reference.

    Ideal programming still leaves DAC/ADC conversion error, non-linearity and read noise,
    so all comparisons are tolerance based.
    """
    lq8, aq, x = _calibrated_pair(8)
    ref_float = torch.nn.functional.linear(x, lq8.weight.data, lq8.bias.data)

    np.random.seed(0)
    mem_uniform8 = _convert(lq8, aq)
    assert isinstance(mem_uniform8, TiledMemristorLinear)

    lq_grid8, _, _ = _calibrated_pair([[8, 8], [8, 8]])
    with torch.no_grad():
        lq_grid8.weight.copy_(lq8.weight), lq_grid8.bias.copy_(lq8.bias)
    lq_grid8.train(); _ = lq_grid8(x); lq_grid8.eval()  # calibrate tile observers on the copied weight
    np.random.seed(0)
    mem_grid8 = _convert(lq_grid8, aq)
    assert isinstance(mem_grid8, MixedPrecisionTiledMemristorLinear)
    assert mem_grid8.initialized

    out_uniform = _seeded_forward(mem_uniform8, x).flatten()
    out_grid = _seeded_forward(mem_grid8, x).flatten()
    ref = ref_float.flatten()

    def rel_err(a, b):
        return ((a - b).norm() / b.norm()).item()

    corr = torch.corrcoef(torch.stack([out_uniform, out_grid]))[0, 1].item()
    e_uni, e_grid = rel_err(out_uniform, ref), rel_err(out_grid, ref)
    print(f"  all-8 grid vs uniform-8: corr={corr:.4f}, rel_err vs float: uniform={e_uni:.4f}, grid={e_grid:.4f}")
    assert corr > 0.98, corr
    assert abs(e_uni - e_grid) < 0.1, (e_uni, e_grid)

    # mixed precision must beat the uniform low precision it contains
    lq4, _, _ = _calibrated_pair(4)
    with torch.no_grad():
        lq4.weight.copy_(lq8.weight), lq4.bias.copy_(lq8.bias)
    lq4.train(); _ = lq4(x); lq4.eval()
    lq_mixed, _, _ = _calibrated_pair([[8, 6], [4, 8]])
    with torch.no_grad():
        lq_mixed.weight.copy_(lq8.weight), lq_mixed.bias.copy_(lq8.bias)
    lq_mixed.train(); _ = lq_mixed(x); lq_mixed.eval()
    np.random.seed(0)
    mem_uniform4 = _convert(lq4, aq)
    np.random.seed(0)
    mem_mixed = _convert(lq_mixed, aq)
    e4 = rel_err(_seeded_forward(mem_uniform4, x).flatten(), ref)
    e_mixed = rel_err(_seeded_forward(mem_mixed, x).flatten(), ref)
    print(f"  rel_err vs float: uniform-4={e4:.4f}, mixed[[8,6],[4,8]]={e_mixed:.4f}, uniform-8={e_uni:.4f}")
    assert e_mixed < e4, (e_mixed, e4)
    print("PASS test_conversion_numerics")


def test_state_dict_parity():
    """Converted wrapper state dict loads strict into the mem_inited factory module."""
    lq, aq, x = _calibrated_pair([[8, 6], [4, 8]])
    np.random.seed(0)
    mem = _convert(lq, aq)
    sd = mem.state_dict()

    fresh = make_inference_memristor_linear(
        in_features=256,
        out_features=256,
        weight_bit_prec=[[8, 6], [4, 8]],
        converter_hardware_settings=HW_SETTINGS,
        name="parity",
    )
    assert isinstance(fresh, MixedPrecisionTiledMemristorLinear)
    assert not fresh.initialized
    fresh.load_state_dict(sd, strict=True)
    fresh.initialized = True
    out_a = _seeded_forward(mem, x, seed=3)
    out_b = _seeded_forward(fresh, x, seed=3)
    assert torch.allclose(out_a, out_b), (out_a - out_b).abs().max()

    # scalar spec keeps producing a plain TiledMemristorLinear
    plain = make_inference_memristor_linear(256, 256, 8, HW_SETTINGS, name="plain")
    assert isinstance(plain, TiledMemristorLinear)
    print("PASS test_state_dict_parity")


def _make_model_config_dict(weight_bit_prec, num_layers=1, conformer_size=128):
    from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000, win_size=0.025, hop_size=0.01, f_min=60, f_max=7600,
        min_amp=1e-10, num_filters=80, center=False,
    )
    frontend_config = cfg_mod.VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32, conv2_channels=64, conv3_channels=64, conv4_channels=32,
        conv_kernel_size=(3, 3), conv_padding=None,
        pool1_kernel_size=(2, 1), pool1_stride=(2, 1), pool1_padding=None,
        pool2_kernel_size=(2, 1), pool2_stride=(2, 1), pool2_padding=None,
        activation_str="ReLU", out_features=conformer_size, activation=None,
    )
    specaug_config = cfg_mod.SpecaugConfig(
        repeat_per_n_frames=25, max_dim_time=20, num_repeat_feat=5, max_dim_feat=16
    )
    pos_emb_config = cfg_mod.ConformerPosEmbConfig(
        learnable_pos_emb=False, rel_pos_clip=16, with_linear_pos=True,
        with_pos_bias=True, separate_pos_emb_per_head=True, pos_emb_dropout=0.0,
    )
    model_config = cfg_mod.QuantModelTrainConfigV16(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        pos_emb_config=pos_emb_config,
        specauc_start_epoch=11,
        label_target_size=44,
        conformer_size=conformer_size,
        num_layers=num_layers,
        num_heads=4,
        ff_dim=2 * conformer_size,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        dropout_broadcast_axes=None,
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor_symmetric",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor_symmetric",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor_symmetric",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor_symmetric",
        moving_average=None,
        weight_bit_prec=weight_bit_prec,
        activation_bit_prec=8,
        quantize_output=False,
        quant_in_linear=True,
        converter_hardware_settings=HW_SETTINGS,
        pos_enc_converter_hardware_settings=None,
        num_cycles=0,
        correction_settings=IDEAL_CORRECTION,
        weight_noise=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
        weight_dropout=0.0,
        weight_pruning=None,
    )
    return asdict(model_config)


def _ensure_run_ctx(stage="train_step", epoch=1):
    from returnn.torch import context as rt_context

    rt_context._run_ctx = rt_context.RunCtx(stage=stage, device="cpu", engine=None, epoch=epoch)


MIXED_LAYER_SPEC = {
    # conformer_size=128, ff_dim=256: lin_1 is 256x128 (2x1 tiles), lin_2 is 128x256 (1x2)
    "ff": {"lin_1": [[8], [6]], "lin_2": [[8, 4]]},
    # W_i is 384x128 (3x1, via 1D shorthand), W_o 128x128, learn_emb 128x128
    "mhsa": {"W_i": [8, 6, 4], "W_o": 4, "learn_emb": 8},
    # pconv_1 is 256x128 (2x1), pconv_2 128x128 (1x1)
    "conv": {"pconv_1": [[8], [4]], "pconv_2": [[6]], "dconv": 8},
}


def _assert_no_stray_weights(state_dict):
    # same check as mem_init_hook in the training network file
    for name in state_dict.keys():
        if "weight" in name and "memristor" not in name:
            if any(x in name for x in ["linear", "conv"]):
                if not any(x in name for x in ["frontend", "final"]):
                    raise AssertionError((name, "There should not be a non memristor weight here"))


def test_full_model():
    """End-to-end: QAT model with mixed spec -> prep_quant -> mem_inited model parity.

    Also covers backward compatibility: legacy config styles must still build and run.
    """
    _ensure_run_ctx()

    # backward compat: legacy config styles still build with the legacy quantizer layout
    for legacy_spec in [8, [8], [{"ff": 4, "mhsa": 8, "conv": 4}]]:
        cfg_dict = _make_model_config_dict(legacy_spec)
        torch.manual_seed(0)
        m = net.Model(model_config_dict=cfg_dict, epoch=1, step=0)
        assert not any(isinstance(mod, TiledWeightQuantizer) for mod in m.modules()), legacy_spec
        assert not any("quantizers" in k for k in m.state_dict().keys()), legacy_spec
    print("  legacy config styles build with unchanged state-dict layout")

    cfg_dict = _make_model_config_dict([MIXED_LAYER_SPEC])
    torch.manual_seed(0)
    model = net.Model(model_config_dict=cfg_dict, epoch=1, step=0)
    n_tiled = sum(isinstance(m, TiledWeightQuantizer) for m in model.modules())
    # 2 ff modules x (lin_1 + lin_2) + W_i + pconv_1 + pconv_2 (1 layer); W_o/learn_emb/dconv scalar
    assert n_tiled == 7, n_tiled
    # W_o stays a plain WeightQuantizer (scalar spec)
    mhsa = model.conformer.module_list[0].module_list[2].mhsa
    assert isinstance(mhsa.out_proj.weight_quantizer, WeightQuantizer)
    assert isinstance(mhsa.qkv_proj.weight_quantizer, TiledWeightQuantizer)

    torch.manual_seed(1)
    raw_audio = torch.randn(2, 16000, 1) * 0.1
    raw_len = torch.tensor([16000, 12000])
    model.train()
    _ = model(raw_audio=raw_audio, raw_audio_len=raw_len)  # calibrates all observers
    model.eval()
    with torch.no_grad():
        qat_logprobs, out_len = model(raw_audio=raw_audio, raw_audio_len=raw_len)
    assert torch.isfinite(qat_logprobs[-1]).all()

    np.random.seed(0)
    torch.manual_seed(0)
    model.prep_quant()
    converted_sd = model.state_dict()
    _assert_no_stray_weights(converted_sd)
    assert any("tiles." in k for k in converted_sd.keys())
    with torch.no_grad():
        torch.manual_seed(2)
        mem_logprobs, _ = model(raw_audio=raw_audio, raw_audio_len=raw_len)
    assert torch.isfinite(mem_logprobs[-1]).all()
    print("  converted model: no stray weights, forward finite")

    # mem_inited variant: construct directly, load the converted checkpoint strict
    torch.manual_seed(0)
    model_mi = net_mem_inited.Model(model_config_dict=cfg_dict, epoch=1, step=0)
    missing, unexpected = model_mi.load_state_dict(converted_sd, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    model_mi.prep_quant()  # only flips .initialized, incl. the mixed wrapper property setter
    model_mi.eval()
    with torch.no_grad():
        torch.manual_seed(2)
        mi_logprobs, _ = model_mi(raw_audio=raw_audio, raw_audio_len=raw_len)
    diff = (mi_logprobs[-1] - mem_logprobs[-1]).abs().max().item()
    assert diff < 1e-5, diff  # same memristor state + same seed -> same outputs
    print(f"  mem_inited parity: max |diff| = {diff:.2e}")
    print("PASS test_full_model")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    test_spec_normalization()
    test_qat_forward_backward()
    test_scalar_equivalence()
    test_legacy_state_dict_layout()
    test_uniform_checkpoint_into_tiled()
    test_noise_per_tile()
    test_conversion_numerics()
    test_state_dict_parity()
    test_full_model()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
