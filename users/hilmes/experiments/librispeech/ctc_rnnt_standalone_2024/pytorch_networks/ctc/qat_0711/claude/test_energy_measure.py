"""Manual verification script for the energy/current/device measurement (energy_measure.py).

Keep this script; it is meant to be re-run manually after changes. Run from the experiment
root (/u/hilmes/experiments/asr_2023):

    PYTHONPATH=recipe:i6_models_full:/u/hilmes/src/MiniReturnn \
        python3 recipe/i6_experiments/users/hilmes/experiments/librispeech/\
ctc_rnnt_standalone_2024/pytorch_networks/ctc/qat_0711/claude/test_energy_measure.py

The script also inserts the paths itself, so a plain `python3 <script>` works too.

Checks:
  1. the synaptogen capture hook is passive: attaching the collector does not change any
     output (bit-exact under the same seed), and the captured differential column currents
     equal the live PairedMemristorArrayV2 outputs exactly
  2. accumulator arithmetic, the per-cell >= column cancellation inequality, ADC
     saturation counting and the raw-sample batch/layer/dtype filters
  3. device statistics: allocated/padding match the structural formulas for uniform and
     mixed-precision linears, zero weights (pruning) drive conducting to 0
  4. full-model end to end with the energy mem_inited subclass: attach, forward,
     finish_batch, report pickle/json round-trip
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve()
_exp_root = _here.parents[11]  # .../experiments/asr_2023
for p in [_exp_root / "recipe", _exp_root / "i6_models_full", Path("/u/hilmes/src/MiniReturnn")]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import json
import pickle
import tempfile
import importlib

import numpy as np
import torch

_pkg = "i6_experiments.users.hilmes.experiments.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.qat_0711.claude"

tmp = importlib.import_module(_pkg + ".test_mixed_prec_tiles")  # shared helpers/configs
em = importlib.import_module(_pkg + ".energy_measure")
net = importlib.import_module(_pkg + ".memristor_v16_dynmic_prec")
net_energy = importlib.import_module(_pkg + ".memristor_v16_dynmic_prec_energy")
net_energy_mi = importlib.import_module(_pkg + ".memristor_v16_dynmic_prec_energy_mem_inited")

from synaptogen_ml.memristor_modules.memristor import MemristorArray, PairedMemristorArrayV2
from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear


def _converted_uniform(spec=4, out_features=256, in_features=256, seed=42):
    lq, aq, x = tmp._calibrated_pair(spec, out_features=out_features, in_features=in_features, seed=seed)
    np.random.seed(0)
    return tmp._convert(lq, aq), x


def _cfg(**overrides):
    """EnergyMeasureConfig has no defaults (all options must be explicit in the hashed
    job config); this is the test-local summary baseline."""
    base = dict(detail="summary", raw_batches=0, raw_layers=[], raw_max_rows=0,
                raw_dtype="float16", percell_map_layers=[], enabled=True)
    base.update(overrides)
    return em.EnergyMeasureConfig(**base)


def _full_cfg(**overrides):
    base = dict(detail="full", raw_batches=1, raw_layers=None, raw_max_rows=128, percell_map_layers=[0])
    base.update(overrides)
    return _cfg(**base)


def test_capture_passive_and_exact():
    """Attaching must not change outputs; captured currents == live pair outputs."""
    mem, x = _converted_uniform(4)
    out_ref = tmp._seeded_forward(mem, x, seed=7)

    collector = em.attach(mem, _full_cfg(raw_dtype="float32"))
    n_arrays = sum(isinstance(m, MemristorArray) for m in mem.modules())
    assert len(collector.infos) == n_arrays, (len(collector.infos), n_arrays)

    out_hooked = tmp._seeded_forward(mem, x, seed=7)
    assert torch.equal(out_ref, out_hooked), "capture hook changed the forward output"
    assert collector.hook_calls == n_arrays, (collector.hook_calls, n_arrays)

    # captured pos - neg must equal the live differential pair outputs, bit exact
    pair_outputs = {}
    for name, mod in mem.named_modules():
        if isinstance(mod, PairedMemristorArrayV2):
            def _wrap(inputs, _orig=mod.forward, _name=name):
                out = _orig(inputs)
                pair_outputs[_name] = out.detach().clone()
                return out

            mod.forward = _wrap
    collector2 = em.attach(mem, _full_cfg(raw_dtype="float32"))
    _ = tmp._seeded_forward(mem, x, seed=7)
    by_path = {info.path: info for info in collector2.infos}
    checked = 0
    for pair_name, pair_out in pair_outputs.items():
        pos = collector2.raw_samples[by_path[pair_name + ".pos"].index][0]["column_current"]
        neg = collector2.raw_samples[by_path[pair_name + ".neg"].index][0]["column_current"]
        assert torch.equal(pos - neg, pair_out), pair_name
        checked += 1
    assert checked == len(mem.memristors), checked
    print("PASS test_capture_passive_and_exact")


def test_accumulators_and_filters():
    mem, x = _converted_uniform(4)
    cfg = _full_cfg(percell_map_layers=None)
    collector = em.attach(mem, cfg)

    _ = tmp._seeded_forward(mem, x, seed=1)
    collector.finish_batch()
    _ = tmp._seeded_forward(mem, x, seed=2)
    collector.finish_batch()
    assert collector.batch_idx == 2

    rows_per_call = x.shape[0]
    by_path = {info.path: info for info in collector.infos}
    for info in collector.infos:
        acc = collector.acc[info.index]
        assert acc["count"] == 2 * rows_per_call, acc["count"]
        assert torch.isfinite(acc["sum"]).all() and torch.isfinite(acc["sum_sq"]).all()
        assert (acc["sum_abs"] >= acc["sum"].abs() - 1e-12).all()
        assert (acc["max_abs"] > 0).any()
        # cancellation inequality: per-cell |I| >= |column current| in aggregate
        assert acc["sum_abs_percell"] >= float(acc["sum_abs"].sum().item()) - 1e-12
        assert acc["sum_abs_iv"] >= 0.0
        assert info.adc_clip_current > 0

    # differential pair accumulators: linearity vs the per-polarity sums, triangle inequality
    for pair_path, pacc in collector.pair_acc.items():
        pos_acc = collector.acc[by_path[pair_path + ".pos"].index]
        neg_acc = collector.acc[by_path[pair_path + ".neg"].index]
        assert pacc["count"] == 2 * rows_per_call
        assert torch.allclose(pacc["sum"], pos_acc["sum"] - neg_acc["sum"], atol=1e-9)
        assert (pacc["sum_abs"] <= pos_acc["sum_abs"] + neg_acc["sum_abs"] + 1e-9).all()
        assert (pacc["max_abs"] > 0).any()

    # raw_max_rows caps the stored (batch x time) rows per sample
    mem_cap, x_cap = _converted_uniform(4)
    collector_cap = em.attach(mem_cap, _full_cfg(raw_max_rows=4))
    _ = tmp._seeded_forward(mem_cap, x_cap, seed=5)
    collector_cap.finish_batch()
    for samples in collector_cap.raw_samples.values():
        assert samples[0]["column_current"].shape[0] == 4, samples[0]["column_current"].shape

    # saturation counting: clip forced to 0 -> everything saturates; clip inf -> nothing
    mem_sat, x_sat = _converted_uniform(4)
    collector_sat = em.attach(mem_sat, _cfg())
    pair_paths = sorted(collector_sat.pair_infos.keys())
    collector_sat.pair_infos[pair_paths[0]]["adc_clip_current"] = 0.0
    collector_sat.pair_infos[pair_paths[1]]["adc_clip_current"] = float("inf")
    _ = tmp._seeded_forward(mem_sat, x_sat, seed=3)
    collector_sat.finish_batch()
    pacc0 = collector_sat.pair_acc[pair_paths[0]]
    n_reads = pacc0["count"] * collector_sat.pair_infos[pair_paths[0]]["out_features"]
    assert pacc0["sat_count"] >= 0.99 * n_reads, (pacc0["sat_count"], n_reads)
    assert collector_sat.pair_acc[pair_paths[1]]["sat_count"] == 0

    # raw samples: only batch 0 (raw_batches=1), stored in fp16
    assert set(collector.raw_samples.keys()) == {i.index for i in collector.infos}
    for samples in collector.raw_samples.values():
        assert len(samples) == 1 and samples[0]["batch"] == 0
        assert samples[0]["column_current"].dtype == torch.float16
    # per-cell maps for all arrays (filter None), correct shape
    modules = dict(mem.named_modules())
    for info in collector.infos:
        assert tuple(collector.percell_maps[info.index].shape) == tuple(modules[info.path].r.shape)

    # peaks were folded per batch
    assert collector.peaks and all(v > 0 for v in collector.peaks.values())
    print("PASS test_accumulators_and_filters")


def test_summary_mode():
    """Summary-mode numbers equal the aggregates of a full run; detail containers stay empty."""
    mem_full, x = _converted_uniform(4)
    mem_sum, _ = _converted_uniform(4)  # same seeds -> identical memristor state
    col_full = em.attach(mem_full, _full_cfg())
    col_sum = em.attach(mem_sum, _cfg())  # summary baseline
    assert col_sum.config.detail == "summary"

    _ = tmp._seeded_forward(mem_full, x, seed=11)
    _ = tmp._seeded_forward(mem_sum, x, seed=11)
    col_full.finish_batch()
    col_sum.finish_batch()

    for info_f, info_s in zip(col_full.infos, col_sum.infos):
        assert info_f.path == info_s.path
        acc_f, acc_s = col_full.acc[info_f.index], col_sum.acc[info_s.index]
        assert "sum" not in acc_s and "sum_sq" not in acc_s
        assert acc_s["count"] == acc_f["count"]
        assert torch.allclose(acc_s["sum_abs"], acc_f["sum_abs"].sum(), atol=1e-12)
        assert torch.equal(acc_s["max_abs"], acc_f["max_abs"].max())
        assert torch.allclose(acc_s["sum_abs_percell"], acc_f["sum_abs_percell"])
        assert torch.allclose(acc_s["sum_abs_iv"], acc_f["sum_abs_iv"])
    for pair_path in col_full.pair_acc:
        pacc_f, pacc_s = col_full.pair_acc[pair_path], col_sum.pair_acc[pair_path]
        assert torch.allclose(pacc_s["sum_abs"], pacc_f["sum_abs"].sum(), atol=1e-12)
        assert int(pacc_s["sat_count"].item()) == int(pacc_f["sat_count"].item())

    # detail containers stay empty; report shrinks to a few MB and round-trips
    assert not col_sum.raw_samples and not col_sum.percell_maps and not col_sum.peaks
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = str(Path(tmpdir) / "energy_report.pkl")
        json_path = str(Path(tmpdir) / "energy_summary.json")
        col_sum.write_report(pkl_path, json_path)
        size_mb = Path(pkl_path).stat().st_size / 1e6
        with open(pkl_path, "rb") as f:
            pickle.load(f)
        with open(json_path) as f:
            summary = json.load(f)
    full_summary = col_full._summary()
    for key, e in summary["per_layer_matrix"].items():
        for metric in ["charge_proxy_sum_abs", "sum_abs_iv", "sum_abs_percell",
                       "max_bitline_current", "max_diff_column_current", "sat_count", "column_reads"]:
            ref = full_summary["per_layer_matrix"][key][metric]
            assert abs(e[metric] - ref) <= 1e-9 * max(1.0, abs(ref)), (key, metric, e[metric], ref)
    assert summary["peak_upper_bound"] == 0.0
    print(f"PASS test_summary_mode (summary pickle {size_mb:.2f} MB)")


def test_device_stats():
    # uniform with padding: 300x200 (out x in), wp=4 -> it=2, ot=3, 18 pairs
    mem, _x = _converted_uniform(4, out_features=300, in_features=200)
    collector = em.attach(mem, _cfg())
    totals = collector.device_stats["totals"]
    wp, it, ot = 4, 2, 3
    assert totals["num_arrays"] == 2 * (wp - 1) * it * ot
    assert totals["allocated"] == 2 * (wp - 1) * it * ot * 128 * 128
    assert totals["padding"] == 2 * (wp - 1) * (it * ot * 128 * 128 - 200 * 300)
    assert 0 < totals["conducting"] <= totals["allocated"] - totals["padding"]

    # mixed precision: per-tile precisions and allocation
    lq, aq, x = tmp._calibrated_pair([[8, 6], [4, 8]], out_features=256, in_features=256)
    np.random.seed(0)
    mixed = tmp._convert(lq, aq)
    collector_mixed = em.attach(mixed, _cfg())
    stats = collector_mixed.device_stats["per_matrix"]
    assert len(stats) == 1
    entry = next(iter(stats.values()))
    assert entry["weight_precisions"] == {"(0, 0)": 8, "(0, 1)": 6, "(1, 0)": 4, "(1, 1)": 8}
    expected_alloc = sum(2 * (p - 1) * 128 * 128 for p in [8, 6, 4, 8])
    assert entry["allocated"] == expected_alloc, (entry["allocated"], expected_alloc)
    assert entry["padding"] == 0

    # all-zero weights (= fully pruned) -> every cell in HRS, nothing conducts
    lq0, aq0, _ = tmp._calibrated_pair(4, out_features=128, in_features=128)
    lq0.weight.data.zero_()
    np.random.seed(0)
    mem0 = tmp._convert(lq0, aq0)
    collector0 = em.attach(mem0, _cfg())
    assert collector0.device_stats["totals"]["conducting"] == 0
    print("PASS test_device_stats")


def test_full_model_energy():
    tmp._ensure_run_ctx()
    assert getattr(net_energy.Model, "is_energy_measure_variant", False)
    assert issubclass(net_energy.Model, net.Model)

    cfg_dict = tmp._make_model_config_dict([tmp.MIXED_LAYER_SPEC])
    torch.manual_seed(0)
    model = net.Model(model_config_dict=cfg_dict, epoch=1, step=0)
    torch.manual_seed(1)
    raw_audio = torch.randn(2, 16000, 1) * 0.1
    raw_len = torch.tensor([16000, 12000])
    model.train()
    _ = model(raw_audio=raw_audio, raw_audio_len=raw_len)
    model.eval()
    np.random.seed(0)
    torch.manual_seed(0)
    model.prep_quant()
    converted_sd = model.state_dict()

    torch.manual_seed(0)
    model_energy = net_energy_mi.Model(model_config_dict=cfg_dict, epoch=1, step=0)
    assert getattr(model_energy, "is_energy_measure_variant", False)
    missing, unexpected = model_energy.load_state_dict(converted_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model_energy.prep_quant()
    model_energy.eval()

    collector = em.attach(model_energy, _full_cfg(raw_layers=[0]))
    n_arrays = sum(isinstance(m, MemristorArray) for m in model_energy.modules())
    assert len(collector.infos) == n_arrays and n_arrays > 0

    with torch.no_grad():
        torch.manual_seed(2)
        logprobs, _ = model_energy(raw_audio=raw_audio, raw_audio_len=raw_len)
    assert torch.isfinite(logprobs[-1]).all()
    collector.finish_batch()
    assert collector.hook_calls == n_arrays, (collector.hook_calls, n_arrays)

    # metadata sanity on the parsed tree; lin_out stays a plain nn.Linear here because the
    # test config uses quantize_output=False, so no arrays are expected for it
    matrices = {(i.layer, i.matrix) for i in collector.infos}
    for expected in [(0, "lin_1"), (0, "lin_2"), (0, "W_i"), (0, "W_o"), (0, "learn_emb"),
                     (0, "pconv_1"), (0, "pconv_2"), (0, "dconv")]:
        assert expected in matrices, (expected, sorted(matrices))
    assert matrices == {(0, m) for m in
                        ["lin_1", "lin_2", "W_i", "W_o", "learn_emb", "pconv_1", "pconv_2", "dconv"]}
    w_i = [i for i in collector.infos if i.matrix == "W_i"]
    assert {i.tile for i in w_i} == {(0, 0), (1, 0), (2, 0)}  # 384x128 -> 3x1 tile grid
    assert {i.weight_precision for i in w_i} == {8, 6, 4}
    # raw filter: layer 0 captured, lin_out (layer -1) not
    raw_layers = {collector.infos[idx].layer for idx in collector.raw_samples}
    assert raw_layers == {0}, raw_layers

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = str(Path(tmpdir) / "energy_report.pkl")
        json_path = str(Path(tmpdir) / "energy_summary.json")
        collector.write_report(pkl_path, json_path)
        with open(pkl_path, "rb") as f:
            report = pickle.load(f)
        with open(json_path) as f:
            summary = json.load(f)
    assert report["device_stats"]["totals"] == summary["devices"]
    assert summary["num_batches"] == 1 and summary["hook_calls"] == n_arrays
    assert summary["current_totals"]["charge_proxy_sum_abs"] > 0
    assert summary["peak_upper_bound"] > 0 and report["peaks"]
    assert len(report["accumulators"]) == n_arrays
    assert len(report["pair_accumulators"]) == n_arrays // 2
    assert all(isinstance(a["sum_abs"], np.ndarray) for a in report["accumulators"])
    print(
        f"  devices: allocated={summary['devices']['allocated']}, "
        f"conducting={summary['devices']['conducting']}, padding={summary['devices']['padding']}"
    )
    print(f"  charge proxy total: {summary['current_totals']['charge_proxy_sum_abs']:.3e} A summed over reads")
    print("PASS test_full_model_energy")


def main():
    test_capture_passive_and_exact()
    test_accumulators_and_filters()
    test_summary_mode()
    test_device_stats()
    test_full_model_energy()
    print("\nALL ENERGY MEASURE TESTS PASSED")


if __name__ == "__main__":
    main()
