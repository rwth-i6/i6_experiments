import re
from typing import Any, Dict, List, Optional, Union

from sisyphus import tk, Job, Task

from i6_core.returnn.training import PtCheckpoint


def _normalize_grid(spec: Any, out_features: int, in_features: int, tile_size: int) -> List[List[int]]:
    """Normalize a precision spec to a per-tile 2D grid for a weight [out, in].

    Semantics mirror memristor_v16_dynmic_prec_cfg.SubMatrixPrecision.from_spec: scalar =
    uniform precision, 1D list = one precision per output-row tile broadcast across the
    input tiles, 2D list = full per-tile grid. Only integer precisions are supported.
    """
    num_row_tiles = -(-out_features // tile_size)
    num_col_tiles = -(-in_features // tile_size)
    if isinstance(spec, int):
        grid = [[spec] * num_col_tiles for _ in range(num_row_tiles)]
    elif isinstance(spec[0], (list, tuple)):
        assert len(spec) == num_row_tiles and all(len(row) == num_col_tiles for row in spec), (
            f"2D spec does not match {num_row_tiles}x{num_col_tiles} tiles of [{out_features}, {in_features}]"
        )
        grid = [list(row) for row in spec]
    else:
        assert len(spec) == num_row_tiles, f"1D spec length {len(spec)} != {num_row_tiles} output row tiles"
        grid = [[prec] * num_col_tiles for prec in spec]
    assert all(isinstance(p, int) and 2 <= p <= 8 for row in grid for p in row), grid
    return grid


class CalculateMemristorDeviceStatsJob(Job):
    """
    Computes how many memristor devices a quantized model occupies: 128x128 crossbar
    arrays and individual resistances (cells), following the synaptogen_ml hardware
    mapping. A weight [out, in] is split into tile_size x tile_size tiles; a tile at
    integer precision B needs B-1 magnitude bit-planes (the sign lives in the pos/neg
    split), each a pair of full crossbars -- 2*(B-1) crossbars / 2*(B-1)*tile_size^2
    resistances per tile, edge tiles zero-padded. The depthwise conv maps to structural
    (non-crossbar) arrays with 2*(B-1)*channels*kernel cells.

    Utilization (fraction of cells programmed nonzero) re-quantizes the raw weights
    per tensor (symmetric, scale from the weight quantizer observer min/max in the
    checkpoint if present, else the tensor max-abs) -- a close approximation of the
    trained quantizers.
    """

    # the conformer-block matrices converted to memristors (same selection as the
    # pruning stats registration); frontend and final linear stay digital
    DEFAULT_WEIGHT_NAME_PATTERNS = [
        r"linear_ff\.weight$",
        r"linear_out\.weight$",
        r"pointwise_conv1\.weight$",
        r"pointwise_conv2\.weight$",
        r"qkv_proj\.weight$",
        r"out_proj\.weight$",
        r"linear_pos\.weight$",
    ]
    DEFAULT_DCONV_NAME_PATTERNS = [r"depthwise_conv\.weight$"]

    def __init__(
        self,
        checkpoint: Union[PtCheckpoint, tk.Path],
        weight_bit_prec: Union[int, Dict[str, Any]] = 8,
        tile_size: int = 128,
        weight_name_patterns: Optional[List[str]] = None,
        dconv_name_patterns: Optional[List[str]] = None,
        compute_utilization: bool = True,
    ):
        """
        :param checkpoint: checkpoint to analyze
        :param weight_bit_prec: single integer precision for all matched tensors, or a
            dict mapping regex pattern -> spec (scalar / 1D per-output-row-tile list /
            2D per-tile grid, v16 SubMatrixPrecision semantics); first match wins
        :param tile_size: memristor crossbar edge length
        :param weight_name_patterns: regexes selecting the memristor linear weights
        :param dconv_name_patterns: regexes selecting depthwise conv weights
            ([channels, 1, kernel]); their spec must be scalar
        :param compute_utilization: also count cells programmed nonzero
        """
        self.checkpoint = checkpoint
        self.weight_bit_prec = weight_bit_prec
        self.tile_size = tile_size
        self.weight_name_patterns = (
            weight_name_patterns if weight_name_patterns is not None else self.DEFAULT_WEIGHT_NAME_PATTERNS
        )
        self.dconv_name_patterns = (
            dconv_name_patterns if dconv_name_patterns is not None else self.DEFAULT_DCONV_NAME_PATTERNS
        )
        self.compute_utilization = compute_utilization

        self.out_num_crossbars = self.output_var("num_crossbars")
        self.out_num_resistances = self.output_var("num_resistances")
        self.out_stats = self.output_var("stats")
        self.out_report = self.output_path("device_stats.txt")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def _spec_for(self, name: str) -> Any:
        if isinstance(self.weight_bit_prec, int):
            return self.weight_bit_prec
        for pattern, spec in self.weight_bit_prec.items():
            if re.search(pattern, name):
                return spec
        raise AssertionError(f"no precision spec matches tensor {name}")

    def run(self):
        import torch

        state = torch.load(str(self.checkpoint), map_location=torch.device("cpu"))
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        def observer_scale(name: str, quant_max: int, tile_rc: Optional[tuple] = None) -> Optional[float]:
            # per-tile observers of the v16 mixed precision models take precedence over the
            # per-tensor observer of the uniform models
            prefix = name[: -len(".weight")] + ".weight_quantizer"
            keys = [prefix + ".observer"]
            if tile_rc is not None:
                keys.insert(0, prefix + f".quantizers.{tile_rc[0]}.{tile_rc[1]}.observer")
            for key in keys:
                min_val = state.get(key + ".min_val")
                max_val = state.get(key + ".max_val")
                if min_val is not None and max_val is not None:
                    bound = max(abs(float(min_val)), abs(float(max_val)))
                    if bound > 0:
                        return bound / quant_max
            return None

        CELL_KEYS = ("programmed_cells", "weight_driven_zeros", "sign_zeros", "zero_weight_cells")

        def cell_counts(tile: "torch.Tensor", prec: int, name: str, tile_rc: Optional[tuple] = None) -> tuple:
            """(programmed, weight-driven zeros, sign-line zeros, zero-weight cells).

            Each set bit of the abs quantized integer programs exactly one cell of the
            pos/neg pair of its bit-plane (mirrors TiledMemristorLinear.init_from_linear_quant).
            The unset magnitude bits of a nonzero weight are weight-driven zeros, the whole
            opposite-sign line is zero on top of that (sign zeros), and weights quantized to
            exactly 0 leave both lines empty (zero-weight cells).
            """
            quant_max = 2 ** (prec - 1) - 1
            scale = observer_scale(name, quant_max, tile_rc)
            if scale is None:
                max_abs = float(tile.abs().max())
                scale = max_abs / quant_max if max_abs > 0 else 1.0
            q_abs = torch.clamp(torch.round(tile.abs() / scale), 0, quant_max).to(torch.int64)
            planes = prec - 1
            programmed = sum(int(((q_abs // (2**bit)) % 2 != 0).sum()) for bit in range(planes))
            nonzero = int((q_abs != 0).sum())
            return (
                programmed,
                planes * nonzero - programmed,
                planes * nonzero,
                2 * planes * (q_abs.numel() - nonzero),
            )

        tile_cells = self.tile_size * self.tile_size
        crossbars = 0
        dconv_arrays = 0
        dconv_resistances = 0
        padding_cells = 0
        matched_weights = 0
        cell_totals = dict.fromkeys(CELL_KEYS, 0)
        crossbars_per_precision: Dict[int, int] = {}
        per_tensor: Dict[str, Dict] = {}

        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            is_dconv = any(re.search(p, name) for p in self.dconv_name_patterns)
            if not is_dconv and not any(re.search(p, name) for p in self.weight_name_patterns):
                continue
            tensor = tensor.float()
            spec = self._spec_for(name)
            stats = {"shape": list(tensor.shape), "numel": tensor.numel(), **dict.fromkeys(CELL_KEYS, 0)}

            def add_cells(counts):
                for key, val in zip(CELL_KEYS, counts):
                    stats[key] += val

            if is_dconv:
                assert tensor.dim() == 3 and tensor.shape[1] == 1 and isinstance(spec, int), (name, spec)
                channels, _, kernel = tensor.shape
                stats["crossbars"] = 0
                stats["arrays"] = 2 * (spec - 1)
                stats["resistances"] = stats["arrays"] * channels * kernel
                dconv_arrays += stats["arrays"]
                dconv_resistances += stats["resistances"]
                if self.compute_utilization:
                    add_cells(cell_counts(tensor, spec, name))
            else:
                assert tensor.dim() == 2, (name, tensor.shape)
                out_features, in_features = tensor.shape
                grid = _normalize_grid(spec, out_features, in_features, self.tile_size)
                stats["crossbars"] = 0
                for r, row in enumerate(grid):
                    rows_eff = min(self.tile_size, out_features - r * self.tile_size)
                    for c, prec in enumerate(row):
                        cols_eff = min(self.tile_size, in_features - c * self.tile_size)
                        tile_crossbars = 2 * (prec - 1)
                        stats["crossbars"] += tile_crossbars
                        padding_cells += tile_crossbars * (tile_cells - rows_eff * cols_eff)
                        crossbars_per_precision[prec] = crossbars_per_precision.get(prec, 0) + tile_crossbars
                        if self.compute_utilization:
                            tile = tensor[
                                r * self.tile_size : r * self.tile_size + rows_eff,
                                c * self.tile_size : c * self.tile_size + cols_eff,
                            ]
                            add_cells(cell_counts(tile, prec, name, tile_rc=(r, c)))
                stats["resistances"] = stats["crossbars"] * tile_cells
                crossbars += stats["crossbars"]

            stats["utilization_percent"] = 100.0 * stats["programmed_cells"] / stats["resistances"]
            matched_weights += stats["numel"]
            for key in CELL_KEYS:
                cell_totals[key] += stats[key]
            per_tensor[name] = stats

        assert per_tensor, "no tensors matched the weight selection"
        num_resistances = crossbars * tile_cells + dconv_resistances
        programmed = cell_totals["programmed_cells"]
        if self.compute_utilization:
            # every cell is exactly one of: programmed, weight-driven zero, sign-line zero,
            # zero-weight cell, or edge-tile padding
            assert sum(cell_totals.values()) + padding_cells == num_resistances, (cell_totals, padding_cells)

        self.out_num_crossbars.set(crossbars)
        self.out_num_resistances.set(num_resistances)
        self.out_stats.set(
            {
                "num_crossbars": crossbars,
                "num_dconv_arrays": dconv_arrays,
                "num_resistances": num_resistances,
                "num_dconv_resistances": dconv_resistances,
                "padding_cells": padding_cells,
                "matched_weights": matched_weights,
                "resistances_per_weight": num_resistances / matched_weights,
                **(cell_totals if self.compute_utilization else dict.fromkeys(CELL_KEYS)),
                "utilization_percent": 100.0 * programmed / num_resistances if self.compute_utilization else None,
                "crossbars_per_precision": crossbars_per_precision,
                "per_tensor": per_tensor,
                "tile_size": self.tile_size,
            }
        )

        with open(self.out_report.get_path(), "wt") as f:
            f.write(f"checkpoint: {self.checkpoint}\n")
            f.write(f"crossbars ({self.tile_size}x{self.tile_size}): {crossbars}\n")
            f.write(f"dconv structural arrays: {dconv_arrays}\n")
            f.write(
                f"individual resistances: {num_resistances}"
                f" (crossbar cells {crossbars * tile_cells}, dconv cells {dconv_resistances})\n"
            )
            f.write(f"matched logical weights: {matched_weights}\n")
            f.write(f"resistances per weight: {num_resistances / matched_weights:.2f}\n")
            f.write(f"padding cells (zero-padded edge tiles): {padding_cells}\n")
            if self.compute_utilization:
                f.write(
                    f"programmed (nonzero) cells: {programmed}"
                    f" ({100.0 * programmed / num_resistances:.2f}% utilization, approximate)\n"
                )
                pct = lambda n: 100.0 * n / num_resistances  # noqa: E731
                f.write(
                    f"unprogrammed cells: weight-driven zeros {cell_totals['weight_driven_zeros']}"
                    f" ({pct(cell_totals['weight_driven_zeros']):.2f}%),"
                    f" sign-line zeros {cell_totals['sign_zeros']} ({pct(cell_totals['sign_zeros']):.2f}%),"
                    f" zero-weight cells {cell_totals['zero_weight_cells']}"
                    f" ({pct(cell_totals['zero_weight_cells']):.2f}%)\n"
                )
            f.write("crossbars per precision: ")
            f.write(", ".join(f"{p} bit: {n}" for p, n in sorted(crossbars_per_precision.items())) + "\n\n")
            name_len = max(len(name) for name in per_tensor)
            f.write(f"{'name'.ljust(name_len)}  {'shape'.ljust(16)}  {'xbars':>6}  {'cells':>12}  {'util%':>6}\n")
            for name, stats in per_tensor.items():
                f.write(
                    f"{name.ljust(name_len)}  {str(stats['shape']).ljust(16)}  {stats['crossbars']:>6}"
                    f"  {stats['resistances']:>12}  {stats['utilization_percent']:>6.2f}\n"
                )
