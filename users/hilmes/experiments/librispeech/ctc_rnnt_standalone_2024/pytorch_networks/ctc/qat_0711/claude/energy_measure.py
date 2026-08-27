"""
Instrumentation to measure electrical currents, energy proxies and device usage of
synaptogen_ml memristor modules during inference (recognition).

Relies on the optional ``current_capture_hook`` extension point in
``synaptogen_ml.memristor_modules.memristor.MemristorArray.forward``: when the plain
instance attribute is set, the forward calls
``current_capture_hook(array, voltages, per_cell_currents)`` with the tensors it computes
anyway (voltages ``[...B, I]`` in Volts, per-cell currents ``[...B, ...A, I, O]`` in Amperes,
including read noise). Unset attribute = exact no-op.

Usage (from a decoder init hook, after the checkpoint is loaded / prep_quant ran):

    collector = attach(model, EnergyMeasureConfig(**energy_kwargs))
    ...  # per batch, after the model forward: collector.finish_batch()
    collector.write_report("energy_report.pkl", "energy_summary.json")

Nothing here modifies model behavior; all statistics are reduced from live tensors under
``torch.no_grad()``.
"""

from dataclasses import dataclass, asdict
import json
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


# friendly matrix names matching the SubMatrixPrecision spec keys used in the configs
_MATRIX_NAME_BY_ATTR = {
    "linear_ff": "lin_1",
    "linear_out": "lin_2",
    "qkv_proj": "W_i",
    "out_proj": "W_o",
    "linear_pos": "learn_emb",
    "pointwise_conv1": "pconv_1",
    "pointwise_conv2": "pconv_2",
    "depthwise_conv": "dconv",
    "lin_out": "lin_out",
}

_LAYER_RE = re.compile(r"conformer\.module_list\.(\d+)\.module_list\.(\d+)\.")

_RAW_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass
class EnergyMeasureConfig:
    """
    Options for the measurement; arrives as a plain dict via decoder_args["energy"].

    All fields are deliberately required (no defaults): the complete dict is part of the
    hashed job config, so a job's measurement behavior is always explicit in the sisyphus
    graph and cannot silently change with this file's defaults.

    detail: "summary" keeps only per-array scalar accumulators (device stats, count,
        column-level and per-cell charge proxies, energy proxy, running maxima, ADC
        saturation) - enough to rank models by efficiency, artifact is a few MB. "full"
        additionally keeps per-column vectors, per-cell |I| maps, peak-instantaneous
        groups and raw samples. All sums are additive, so summary numbers equal the
        aggregates of a full run.
    raw_batches: number of leading batches for which raw tensors are stored (full only)
    raw_layers: conformer layer indices for raw capture (-1 = non-conformer arrays such as
        lin_out); None = all layers
    raw_max_rows: cap on stored (batch x time) rows per array and batch, bounds the
        artifact size (a dim-512 model has thousands of arrays)
    raw_dtype: storage dtype of the raw samples ("float16"/"bfloat16"/"float32")
    percell_map_layers: layers for which accumulated per-cell |I| maps are kept (full
        only; full [in, out] resolution - a per-cell time series would be infeasible)
    """

    detail: str
    raw_batches: int
    raw_layers: Optional[List[int]]
    raw_max_rows: int
    raw_dtype: str
    percell_map_layers: Optional[List[int]]
    enabled: bool

    def __post_init__(self):
        assert self.detail in ("summary", "full"), self.detail
        assert self.raw_dtype in _RAW_DTYPES, self.raw_dtype


@dataclass
class ArrayInfo:
    """Static metadata of one MemristorArray instance (one polarity of one paired array)."""

    index: int
    path: str  # full module path of the array
    owner_path: str  # module owning the `memristors` list (tile module for mixed linears)
    group_path: str  # logical weight matrix: the mixed wrapper / uniform linear / conv path
    attr: str  # attribute name of the logical matrix in its parent, e.g. "linear_ff"
    matrix: str  # friendly name (lin_1/W_i/pconv_1/...), fallback = attr
    layer: int  # conformer layer index, -1 outside the conformer blocks
    slot: int  # index inside the block's module_list, -1 outside
    tile: Tuple[int, int]  # (output-row tile, input-col tile) of the 128x128 grid
    bit: int  # bit-slice plane index (MSB first)
    polarity: str  # "pos" | "neg"
    weight_precision: int
    in_features: int  # array input lines (rows), incl. padding
    out_features: int  # array output lines (columns), incl. padding
    padding_cells: int  # allocated cells of this array that map to no weight element
    adc_scaling: float  # hardware_output_current_scaling of the owning converter
    adc_clip_current: float  # |I| in Amperes above which the ADC saturates


def _tensor_dtype(name: str) -> torch.dtype:
    assert name in _RAW_DTYPES, f"unknown raw_dtype {name!r}, use one of {sorted(_RAW_DTYPES)}"
    return _RAW_DTYPES[name]


def _serialize_value(v):
    if isinstance(v, torch.Tensor):
        if v.ndim == 0:
            return v.item()
        return v.detach().cpu().numpy()
    return v


def _parse_layer_slot(path: str) -> Tuple[int, int]:
    m = _LAYER_RE.search(path + ".")
    if m is None:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def _adc_limits(owner) -> Tuple[float, float]:
    hs = getattr(getattr(owner, "converter", None), "hs", None)
    if hs is None:
        return float("nan"), float("inf")
    clip = (hs.adc_max / (2 ** hs.output_precision_bits)) / hs.hardware_output_current_scaling
    return float(hs.hardware_output_current_scaling), float(clip)


def build_array_infos(model: torch.nn.Module) -> List[ArrayInfo]:
    """
    Walk the module tree and derive one ArrayInfo per MemristorArray. Handles the three
    tree shapes:
      uniform linear:  <lin>.memristors.{n}.{pos|neg}, n = bit*(it*ot) + in_tile*ot + out_tile
      mixed linear:    <lin>.tiles.{r}.{c}.memristors.{bit}.{pos|neg} (inner tiling is 1x1)
      depthwise conv:  <conv>.memristors.{bit}.{pos|neg}
    """
    from synaptogen_ml.memristor_modules.memristor import MemristorArray
    from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear

    modules = dict(model.named_modules())
    infos = []
    for path, mod in modules.items():
        if not isinstance(mod, MemristorArray):
            continue
        m = re.match(r"^(?:(.*)\.)?memristors\.(\d+)\.(pos|neg)$", path)
        assert m is not None, f"unexpected MemristorArray path {path}"
        owner_path, pair_idx, polarity = m.group(1) or "", int(m.group(2)), m.group(3)
        owner = modules[owner_path]

        tile_m = re.match(r"^(?:(.*)\.)?tiles\.(\d+)\.(\d+)$", owner_path)
        if tile_m is not None:  # tile of a MixedPrecisionTiledMemristorLinear
            group_path = tile_m.group(1) or ""
            tile = (int(tile_m.group(2)), int(tile_m.group(3)))
            assert isinstance(owner, TiledMemristorLinear) and owner.input_tiling == 1 and owner.output_tiling == 1
            bit = pair_idx
            weight_precision = owner.weight_precision
            padding_cells = mod.r.numel() - owner.in_features * owner.out_features
        elif isinstance(owner, TiledMemristorLinear):
            group_path = owner_path
            it, ot = owner.input_tiling, owner.output_tiling
            bit = pair_idx // (it * ot)
            in_tile = (pair_idx % (it * ot)) // ot
            out_tile = pair_idx % ot
            tile = (out_tile, in_tile)  # (output rows, input cols), matching SubMatrixPrecision
            weight_precision = owner.weight_precision
            rows = max(min(owner.in_features - in_tile * owner.memristor_inputs, owner.memristor_inputs), 0)
            cols = max(min(owner.out_features - out_tile * owner.memristor_outputs, owner.memristor_outputs), 0)
            padding_cells = mod.r.numel() - rows * cols
        else:  # MemristorConv1d (or any other owner with a flat memristors list per bit)
            group_path = owner_path
            tile = (0, 0)
            bit = pair_idx
            weight_precision = getattr(owner, "weight_precision", len(owner.memristors) + 1)
            padding_cells = 0

        attr = group_path.rsplit(".", 1)[-1] if "." in group_path else group_path
        layer, slot = _parse_layer_slot(group_path)
        adc_scaling, adc_clip = _adc_limits(owner)
        infos.append(
            ArrayInfo(
                index=len(infos),
                path=path,
                owner_path=owner_path,
                group_path=group_path,
                attr=attr,
                matrix=_MATRIX_NAME_BY_ATTR.get(attr, attr),
                layer=layer,
                slot=slot,
                tile=tile,
                bit=bit,
                polarity=polarity,
                weight_precision=int(weight_precision),
                in_features=mod.r.shape[-2],
                out_features=mod.r.shape[-1],
                padding_cells=int(padding_cells),
                adc_scaling=adc_scaling,
                adc_clip_current=adc_clip,
            )
        )
    return infos


def compute_device_stats(model: torch.nn.Module, infos: List[ArrayInfo]) -> Dict[str, Any]:
    """
    Static device usage from the programmed cell states, per logical weight matrix:
    allocated (all physical cells incl. differential pairs, bit planes, edge-tile padding),
    conducting (cells in LRS, r < 0.5 -- pruned/zero/padded weights sit in HRS),
    padding (allocated cells that map to no weight element), plus a per-tile breakdown.
    """
    modules = dict(model.named_modules())
    stats: Dict[str, Any] = {}
    for info in infos:
        arr = modules[info.path]
        key = info.group_path
        entry = stats.setdefault(
            key,
            {
                "matrix": info.matrix,
                "layer": info.layer,
                "slot": info.slot,
                "weight_precisions": {},
                "allocated": 0,
                "conducting": 0,
                "padding": 0,
                "tiles": {},
            },
        )
        allocated = arr.r.numel()
        conducting = int((arr.r < 0.5).sum().item())
        pad = info.padding_cells
        entry["allocated"] += allocated
        entry["conducting"] += conducting
        entry["padding"] += pad
        tkey = str(info.tile)
        tile_entry = entry["tiles"].setdefault(
            tkey, {"weight_precision": info.weight_precision, "allocated": 0, "conducting": 0, "padding": 0}
        )
        tile_entry["allocated"] += allocated
        tile_entry["conducting"] += conducting
        tile_entry["padding"] += pad
        entry["weight_precisions"][tkey] = info.weight_precision

    totals = {
        "allocated": sum(e["allocated"] for e in stats.values()),
        "conducting": sum(e["conducting"] for e in stats.values()),
        "padding": sum(e["padding"] for e in stats.values()),
        "num_matrices": len(stats),
        "num_arrays": len(infos),
    }
    return {"per_matrix": stats, "totals": totals}


class EnergyCollector:
    """Holds accumulators and hook closures for one instrumented model."""

    def __init__(self, model: torch.nn.Module, config: EnergyMeasureConfig):
        self.config = config
        self.full = config.detail == "full"
        self.infos = build_array_infos(model)
        assert len(self.infos) > 0, (
            "no MemristorArray modules found - is the model converted/mem_inited "
            "and prep_quant done?"
        )
        self.device_stats = compute_device_stats(model, self.infos)
        self.model_repr = f"{type(model).__module__}.{type(model).__name__}"
        self.model_config_repr = repr(getattr(model, "cfg", None) or getattr(model, "train_config", None))
        self.batch_idx = 0
        self.raw_dtype = _tensor_dtype(config.raw_dtype)

        self.acc: List[Dict[str, Any]] = []
        self.raw_samples: Dict[int, List[Dict[str, Any]]] = {}
        self.percell_maps: Dict[int, torch.Tensor] = {}
        self.hook_calls = 0
        self._peak_batch: Dict[Tuple[int, ...], torch.Tensor] = {}
        self.peaks: Dict[Tuple[int, ...], float] = {}

        # per polarity array: physical bitline currents (supply-side view)
        # per differential pair (pos - neg): what the ADC digitizes, incl. saturation
        self.pair_infos: Dict[str, Dict[str, Any]] = {}
        self.pair_acc: Dict[str, Dict[str, Any]] = {}
        self._pending_pos: Dict[str, torch.Tensor] = {}
        modules = dict(model.named_modules())
        for info in self.infos:
            arr = modules[info.path]
            if self.full:
                self.acc.append(
                    {
                        "count": 0,
                        "sum": torch.zeros(info.out_features, dtype=torch.float64),
                        "sum_abs": torch.zeros(info.out_features, dtype=torch.float64),
                        "sum_sq": torch.zeros(info.out_features, dtype=torch.float64),
                        "max_abs": torch.zeros(info.out_features, dtype=torch.float64),
                        "sum_abs_percell": torch.zeros((), dtype=torch.float64),
                        "sum_abs_iv": torch.zeros((), dtype=torch.float64),
                    }
                )
            else:  # summary: scalar accumulators only (sums over all columns)
                self.acc.append(
                    {
                        "count": 0,
                        "sum_abs": torch.zeros((), dtype=torch.float64),
                        "max_abs": torch.zeros((), dtype=torch.float64),
                        "sum_abs_percell": torch.zeros((), dtype=torch.float64),
                        "sum_abs_iv": torch.zeros((), dtype=torch.float64),
                    }
                )
            if info.polarity == "pos":
                pair_path = info.path[: -len(".pos")]
                self.pair_infos[pair_path] = {
                    "layer": info.layer,
                    "slot": info.slot,
                    "matrix": info.matrix,
                    "tile": info.tile,
                    "bit": info.bit,
                    "group_path": info.group_path,
                    "out_features": info.out_features,
                    "adc_scaling": info.adc_scaling,
                    "adc_clip_current": info.adc_clip_current,
                }
                if self.full:
                    self.pair_acc[pair_path] = {
                        "count": 0,
                        "sum": torch.zeros(info.out_features, dtype=torch.float64),
                        "sum_abs": torch.zeros(info.out_features, dtype=torch.float64),
                        "sum_sq": torch.zeros(info.out_features, dtype=torch.float64),
                        "max_abs": torch.zeros(info.out_features, dtype=torch.float64),
                        "sat_count": torch.zeros((), dtype=torch.long),
                    }
                else:
                    self.pair_acc[pair_path] = {
                        "count": 0,
                        "sum_abs": torch.zeros((), dtype=torch.float64),
                        "max_abs": torch.zeros((), dtype=torch.float64),
                        "sat_count": torch.zeros((), dtype=torch.long),
                    }
            if config.enabled:
                arr.current_capture_hook = self._make_hook(info)

    # ---------------------------------------------------------------- capture

    @staticmethod
    def _to_device(store: Dict[str, Any], device: torch.device):
        for k, v in store.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                store[k] = v.to(device)

    def _make_hook(self, info: ArrayInfo):
        def hook(arr, volts: torch.Tensor, cells: torch.Tensor):
            # all accumulation stays on the compute device; only finish_batch() and
            # write_report() synchronize, so capture adds no per-call device-host traffic
            with torch.no_grad():
                self.hook_calls += 1
                acc = self.acc[info.index]
                self._to_device(acc, cells.device)
                col = cells.sum(dim=-2)  # [...B, ...A, O] - same reduction the forward returns
                flat = col.reshape(-1, col.shape[-1])
                flat_abs = flat.abs()
                acc["count"] += flat.shape[0]
                if self.full:
                    acc["sum"] += flat.sum(dim=0, dtype=torch.float64)
                    acc["sum_abs"] += flat_abs.sum(dim=0, dtype=torch.float64)
                    acc["sum_sq"] += (flat * flat).sum(dim=0, dtype=torch.float64)
                    acc["max_abs"] = torch.maximum(acc["max_abs"], flat_abs.max(dim=0).values.double())
                else:
                    acc["sum_abs"] += flat_abs.sum(dtype=torch.float64)
                    acc["max_abs"] = torch.maximum(acc["max_abs"], flat_abs.max().double())

                # differential pair current = what the ADC digitizes; the pair's forward
                # evaluates pos then neg on the same inputs, so stash pos until neg arrives
                pair_path = info.path[: -len(".pos")]
                if info.polarity == "pos":
                    self._pending_pos[pair_path] = col
                else:
                    pos_col = self._pending_pos.pop(pair_path, None)
                    if pos_col is not None and pos_col.shape == col.shape:
                        pinfo = self.pair_infos[pair_path]
                        pacc = self.pair_acc[pair_path]
                        self._to_device(pacc, cells.device)
                        dflat = (pos_col - col).reshape(-1, col.shape[-1])
                        dabs = dflat.abs()
                        pacc["count"] += dflat.shape[0]
                        if self.full:
                            pacc["sum"] += dflat.sum(dim=0, dtype=torch.float64)
                            pacc["sum_abs"] += dabs.sum(dim=0, dtype=torch.float64)
                            pacc["sum_sq"] += (dflat * dflat).sum(dim=0, dtype=torch.float64)
                            pacc["max_abs"] = torch.maximum(pacc["max_abs"], dabs.max(dim=0).values.double())
                        else:
                            pacc["sum_abs"] += dabs.sum(dtype=torch.float64)
                            pacc["max_abs"] = torch.maximum(pacc["max_abs"], dabs.max().double())
                        pacc["sat_count"] += (dabs > pinfo["adc_clip_current"]).sum()

                # supply-side sums via a fused per-input-row L1 norm over the output lines,
                # avoiding a full-size |cells| copy
                row_l1 = torch.linalg.vector_norm(cells, ord=1, dim=-1)  # [...B, ...A, I]
                acc["sum_abs_percell"] += row_l1.sum(dtype=torch.float64)
                acc["sum_abs_iv"] += (row_l1 * volts.abs()).sum(dtype=torch.float64)

                if not self.full:
                    return

                # per-timestep total |I| of this array: reduce everything except the two
                # leading axes (typically [B, T]); grouped by shape across arrays for the peak
                per_t = col.abs()
                if per_t.ndim > 2:
                    per_t = per_t.sum(dim=tuple(range(2, per_t.ndim)))
                per_t = per_t.double()
                key = tuple(per_t.shape)
                if key in self._peak_batch:
                    self._peak_batch[key] = self._peak_batch[key] + per_t
                else:
                    self._peak_batch[key] = per_t

                in_layer_filter = self.config.percell_map_layers is None or info.layer in self.config.percell_map_layers
                if in_layer_filter:
                    flat_cells = cells.abs().reshape(-1, *arr.r.shape).sum(dim=0, dtype=torch.float64)
                    if info.index in self.percell_maps:
                        self.percell_maps[info.index] = self.percell_maps[info.index].to(flat_cells.device) + flat_cells
                    else:
                        self.percell_maps[info.index] = flat_cells

                raw_layer_ok = self.config.raw_layers is None or info.layer in self.config.raw_layers
                if self.batch_idx < self.config.raw_batches and raw_layer_ok:
                    cap = self.config.raw_max_rows
                    self.raw_samples.setdefault(info.index, []).append(
                        {
                            "batch": self.batch_idx,
                            "shape": tuple(col.shape),
                            "voltages": volts.reshape(-1, volts.shape[-1])[:cap].detach().to(self.raw_dtype).cpu(),
                            "column_current": flat[:cap].detach().to(self.raw_dtype).cpu(),
                        }
                    )

        return hook

    def finish_batch(self):
        """Fold the per-batch peak buffers into running peaks; call once after each model forward."""
        for key, buf in self._peak_batch.items():
            peak = float(buf.max().item())
            self.peaks[key] = max(self.peaks.get(key, 0.0), peak)
        self._peak_batch = {}
        self._pending_pos = {}
        self.batch_idx += 1

    # ---------------------------------------------------------------- reporting

    def _summary(self) -> Dict[str, Any]:
        per_layer_matrix: Dict[str, Dict[str, Any]] = {}
        for info in self.infos:
            acc = self.acc[info.index]
            key = f"layer{info.layer}.slot{info.slot}.{info.matrix}"
            e = per_layer_matrix.setdefault(
                key,
                {
                    "charge_proxy_sum_abs": 0.0,
                    "sum_abs_iv": 0.0,
                    "sum_abs_percell": 0.0,
                    "max_bitline_current": 0.0,
                    "max_diff_column_current": 0.0,
                    "column_reads": 0,
                    "sat_count": 0,
                },
            )
            e["charge_proxy_sum_abs"] += float(acc["sum_abs"].sum().item())
            e["sum_abs_iv"] += float(acc["sum_abs_iv"].item())
            e["sum_abs_percell"] += float(acc["sum_abs_percell"].item())
            e["max_bitline_current"] = max(e["max_bitline_current"], float(acc["max_abs"].max().item()))
        for pair_path, pacc in self.pair_acc.items():
            pinfo = self.pair_infos[pair_path]
            key = f"layer{pinfo['layer']}.slot{pinfo['slot']}.{pinfo['matrix']}"
            e = per_layer_matrix[key]
            e["max_diff_column_current"] = max(e["max_diff_column_current"], float(pacc["max_abs"].max().item()))
            e["column_reads"] += pacc["count"] * pinfo["out_features"]
            e["sat_count"] += int(pacc["sat_count"].item())
        for e in per_layer_matrix.values():
            e["sat_fraction"] = e["sat_count"] / e["column_reads"] if e["column_reads"] else 0.0
        dev = self.device_stats
        return {
            "model": self.model_repr,
            "num_batches": self.batch_idx,
            "hook_calls": self.hook_calls,
            "devices": dev["totals"],
            "devices_per_matrix": {
                k: {kk: v[kk] for kk in ("matrix", "layer", "allocated", "conducting", "padding")}
                for k, v in dev["per_matrix"].items()
            },
            "current_totals": {
                "charge_proxy_sum_abs": sum(e["charge_proxy_sum_abs"] for e in per_layer_matrix.values()),
                "sum_abs_iv": sum(e["sum_abs_iv"] for e in per_layer_matrix.values()),
                "sum_abs_percell": sum(e["sum_abs_percell"] for e in per_layer_matrix.values()),
            },
            "peak_groups": {str(k): v for k, v in self.peaks.items()},
            "peak_upper_bound": sum(self.peaks.values()),
            "per_layer_matrix": per_layer_matrix,
        }

    def report_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model_repr,
                "model_config": self.model_config_repr,
                "measure_config": asdict(self.config),
                "num_batches": self.batch_idx,
                "hook_calls": self.hook_calls,
                "units": "currents in Amperes, voltages in Volts; *_iv in Watts summed over reads",
            },
            "device_stats": self.device_stats,
            "arrays": [asdict(info) for info in self.infos],
            "accumulators": [{k: _serialize_value(v) for k, v in acc.items()} for acc in self.acc],
            "pair_infos": self.pair_infos,
            "pair_accumulators": {
                path: {k: _serialize_value(v) for k, v in pacc.items()} for path, pacc in self.pair_acc.items()
            },
            "peaks": {str(k): v for k, v in self.peaks.items()},
            "percell_maps": {
                idx: t.detach().cpu().to(torch.float32).numpy() for idx, t in self.percell_maps.items()
            },
            "raw_samples": {
                idx: [
                    {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
                    for sample in samples
                ]
                for idx, samples in self.raw_samples.items()
            },
            "summary": self._summary(),
        }

    def write_report(self, pickle_path: str, json_path: Optional[str] = None):
        report = self.report_dict()
        with open(pickle_path, "wb") as f:
            pickle.dump(report, f)
        if json_path is not None:
            with open(json_path, "w") as f:
                json.dump(report["summary"], f, indent=2, sort_keys=True)
        print(
            f"[energy_measure] wrote {pickle_path}: {len(self.infos)} arrays, "
            f"{self.batch_idx} batches, devices allocated={self.device_stats['totals']['allocated']}, "
            f"conducting={self.device_stats['totals']['conducting']}"
        )


def attach(model: torch.nn.Module, config: EnergyMeasureConfig) -> EnergyCollector:
    """Instrument all MemristorArrays of the (converted/mem_inited) model; returns the collector."""
    return EnergyCollector(model, config)


def _poly_eval(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Horner evaluation, coefficients in ascending order of degree (the storage
    convention of MemristorArray.resistance_weighted_poly_*). Local copy so the
    collector does not depend on the pin exposing poly_mul_horner."""
    result = torch.zeros_like(x)
    for coeff in coefficients.flip(-1):
        result = result * x + coeff
    return result


class FastEnergyCollector(EnergyCollector):
    """
    Summary-mode collector compatible with the SynaptogenML fast inference path.

    The fast path fuses the per-cell polynomial evaluation and the I-axis reduction
    into one kernel, so the per-cell current tensor the eager
    ``current_capture_hook`` receives never exists. This collector therefore taps
    two other sources, wrapping each array's ``forward`` (instance-attribute
    override; nn.Module forward hooks cannot be used because
    ``PairedMemristorArrayV2.forward`` calls ``self.pos.forward(...)`` directly,
    which bypasses ``__call__``):

    - column statistics (charge proxy, max bitline current, differential-pair /
      ADC-saturation stats) come from the array OUTPUT, which is exactly the
      ``cells.sum(dim=-2)`` reduction the eager hook computes -- identical up to
      the documented ~1e-6 fused-arithmetic drift.
    - the per-cell supply sums (``sum_abs_percell``, ``sum_abs_iv``) are computed
      analytically: per cell I = a_i*(1-r) + b_i*r with a_i/b_i the LLRS/HHRS
      polynomial values of the row voltage. Both have the same sign, so
      sum_o |I_io| = |a_i*(O - R_i) + b_i*R_i| with the STATIC per-row state sum
      R_i = sum_o r_io (precomputed once). Cost O(B*I) instead of O(B*I*O).

    Approximation: the eager sums include the readout-noise sample inside each
    |I|; here noise enters only via the column currents. For read currents >> the
    noise sigma the difference to E[|I+eps|] is negligible -- quantify once
    against an eager reference run (the *_energy_ab_* jobs do exactly that).

    Only ``detail="summary"`` is supported: full mode's per-cell maps and raw
    per-cell samples are exactly what the fast path avoids materializing.
    Works with the standard (non-energy) pins: no ``current_capture_hook``
    extension point is needed.
    """

    def __init__(self, model: torch.nn.Module, config: EnergyMeasureConfig):
        assert config.detail == "summary", (
            "FastEnergyCollector supports detail='summary' only; use the eager "
            "energy pin (import_memristor='energy') for full per-cell detail"
        )
        # enabled=False keeps the base class from attaching per-cell capture hooks
        # (pointless here); restore the caller's config afterwards.
        base_cfg = EnergyMeasureConfig(**{**asdict(config), "enabled": False})
        super().__init__(model, base_cfg)
        self.config = config
        self._row_r_sums: Dict[int, torch.Tensor] = {}
        modules = dict(model.named_modules())
        if config.enabled:
            for info in self.infos:
                arr = modules[info.path]
                # static during recognition: per-input-row sum of the cell states
                self._row_r_sums[info.index] = arr.r.sum(dim=-1)
                self._wrap_array(arr, info)

    def _wrap_array(self, arr: torch.nn.Module, info: ArrayInfo):
        original_forward = arr.forward

        def wrapped_forward(inputs: torch.Tensor, _orig=original_forward, _info=info, _arr=arr):
            col = _orig(inputs)
            self._fast_capture(_arr, _info, inputs, col)
            return col

        arr.forward = wrapped_forward

    def _fast_capture(self, arr, info: ArrayInfo, volts: torch.Tensor, col: torch.Tensor):
        with torch.no_grad():
            self.hook_calls += 1
            acc = self.acc[info.index]
            self._to_device(acc, col.device)
            flat = col.reshape(-1, col.shape[-1])
            flat_abs = flat.abs()
            acc["count"] += flat.shape[0]
            acc["sum_abs"] += flat_abs.sum(dtype=torch.float64)
            acc["max_abs"] = torch.maximum(acc["max_abs"], flat_abs.max().double())

            # differential pair current, same pending-pos mechanism as the eager hook
            # (PairedMemristorArrayV2.forward evaluates pos then neg on the same inputs)
            pair_path = info.path[: -len(".pos")] if info.polarity == "pos" else info.path[: -len(".neg")]
            if info.polarity == "pos":
                self._pending_pos[pair_path] = col
            else:
                pos_col = self._pending_pos.pop(pair_path, None)
                if pos_col is not None and pos_col.shape == col.shape:
                    pinfo = self.pair_infos[pair_path]
                    pacc = self.pair_acc[pair_path]
                    self._to_device(pacc, col.device)
                    dflat = (pos_col - col).reshape(-1, col.shape[-1])
                    dabs = dflat.abs()
                    pacc["count"] += dflat.shape[0]
                    pacc["sum_abs"] += dabs.sum(dtype=torch.float64)
                    pacc["max_abs"] = torch.maximum(pacc["max_abs"], dabs.max().double())
                    pacc["sat_count"] += (dabs > pinfo["adc_clip_current"]).sum()

            # analytic per-row L1 of the cell currents (noise-free), see class docstring
            a = _poly_eval(arr.resistance_weighted_poly_low, volts)
            b = _poly_eval(arr.resistance_weighted_poly_high, volts)
            row_r = self._row_r_sums[info.index]
            if row_r.device != volts.device:
                row_r = row_r.to(volts.device)
                self._row_r_sums[info.index] = row_r
            out_features = arr.r.shape[-1]
            row_l1 = (a * (out_features - row_r) + b * row_r).abs()
            acc["sum_abs_percell"] += row_l1.sum(dtype=torch.float64)
            acc["sum_abs_iv"] += (row_l1 * volts.abs()).sum(dtype=torch.float64)


def attach_fast(model: torch.nn.Module, config: EnergyMeasureConfig) -> FastEnergyCollector:
    """Instrument for the fast inference path (summary mode, standard pins)."""
    return FastEnergyCollector(model, config)
