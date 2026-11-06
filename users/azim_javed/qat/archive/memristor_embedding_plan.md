# Plan: `TiledMemristorEmbedding` for the `_pred` QAT model

## Goal

`FFNNTransducerQATEncoderPredictionModel` uses an `EmbeddingQuant` for its token
embedding (recipe `model_pipelines/common/memristor_layers.py:221`). Every `LinearQuant` in the
model gets converted to a real `TiledMemristorLinear` by `prep_quant`, but there
is no memristor equivalent for the embedding, so the previous suggestion was to
"freeze" it (quantize the weights in software, never touch a memristor array).
That is inconsistent. This plan adds a proper memristor embedding module to
`synaptogen_ml` and wires it into `prep_quant`.

## Why not freeze

Freezing (`weight_quantizer.set_scale_and_zp()` only) keeps `F.embedding` in
software with quantized weights. The embedding never sees DAC/ADC quantization,
readout noise, or programming-cycle drift — so the simulation under-counts the
hardware noise that every other layer experiences. Since the model deliberately
configures `weight_bit_prec` / `weight_quant_method` on the embedding, the
intent is clearly to treat it as hardware.

## Hardware model

`F.embedding(idx, W)` with `W` `[V, D]` is a **row select**: it returns `W[idx]`.
The hardware-faithful analog programs the transposed table into tiled paired
memristor crossbar arrays (`V` input lines, `D` output lines) and, on forward,
**address-decodes** `idx`: drive the single selected input line at DAC full
scale, leave all other lines idle, read out via ADC. Only the input tile
containing the active index is read; the rest stay idle — matching a real
address-decoded ROM and avoiding the wasteful full `V`-wide one-hot matmul.

This reuses the exact bit-slicing / tiling / `DacAdcPair` / `PairedMemristorArrayV2`
machinery from `TiledMemristorLinear`, so the noise and programming model are
identical to the linear layers.

### Scaling convention

`TiledMemristorLinear` sets:
- `input_factor = 1 / (act_scale * act_quant_max)`  (maps quantized activation to `[-1, 1]` for the DAC)
- `output_factor = weight_scale * act_scale * act_quant_max`  (rescales integer readout to real units)

For an embedding the "input" is a discrete index whose active line is driven at
DAC full scale (value `1.0`), i.e. the equivalent `act_scale * act_quant_max = 1.0`.
Therefore:
- `input_factor = 1.0`
- `output_factor = weight_scale`

This makes `dac(1.0 * input_factor)` quantize `1.0` to `dac_max` and output
`+vmax` on the selected wordline — exactly address-decoded ROM behavior.

## Files to touch

1. **NEW** `/work/smt4/azim.javed/repositories/SynaptogenML/synaptogen_ml/memristor_modules/embedding.py`
2. **EDIT** `/work/smt4/azim.javed/repositories/SynaptogenML/synaptogen_ml/memristor_modules/__init__.py`
   (add `from .embedding import *`)
3. **EDIT** `/u/azim.javed/experiments/training/qat/recipe/model_pipelines/ffnn_transducer_qat_encoder_pred/pytorch_modules.py`
   (stash hardware settings in `__init__`; add `prep_quant` + helpers)

`EmbeddingQuant` is **not** added anywhere — it already exists in the recipe at
`model_pipelines/common/memristor_layers.py:221` and is imported by the model.

---

## References (preexisting code to mirror)

All paths are repo-relative. `synaptogen_ml/...` lives under
`/work/smt4/azim.javed/repositories/SynaptogenML/`; `recipe/...` lives under
`/u/azim.javed/experiments/training/qat/recipe/model_pipelines/`.

### For `TiledMemristorEmbedding` (File 1)

`EmbeddingQuant` is structurally a `LinearQuant` over a transposed weight table,
so the new module is a thin specialization of `TiledMemristorLinear`. Keep the
construction, tiling, bit-slicing, programming, and readout loops identical;
only the input path (one-hot address decode instead of a dense activation) and
the scaling (`input_factor=1.0`, `output_factor=weight_scale`) differ.

- `synaptogen_ml/memristor_modules/linear.py:126-163` — `TiledMemristorLinear.__init__`.
  Copy verbatim; rename `in_features`→`num_embeddings`, `out_features`→`embedding_dim`,
  drop `bias` (embeddings have no bias), keep `memristors`/`converter`/`input_factor`/
  `output_factor`/`initialized` exactly.
- `synaptogen_ml/memristor_modules/linear.py:165-170` — `get_memristor_index`. Copy verbatim.
- `synaptogen_ml/memristor_modules/linear.py:172-319` — `init_from_linear_quant`.
  Template for `init_from_embedding_quant`. Differences:
    - No `activation_quant` argument → `input_factor = 1.0`, `output_factor = weight_scale`
      (instead of `weight_scale * act_scale * act_quant_max`).
    - `EmbeddingQuant.weight` is `[V, D]` = `[in=V, out=D]` (row-major, `V` = vocab),
      whereas `LinearQuant.weight` is `[out, in]` and `init_from_linear_quant`
      transposes it to `[in, out]` mid-body. `init_from_embedding_quant` skips that
      inner transpose (the weight is already `[in, out]`); the rest of the body is
      byte-for-byte identical.
- `synaptogen_ml/memristor_modules/linear.py:234-299` — the `ideal_programming` branch
  and the `num_cycles_init * 15` write + `correction_settings` write-verify loop.
  Copy verbatim into `init_from_embedding_quant`.
- `synaptogen_ml/memristor_modules/linear.py:321-358` — `TiledMemristorLinear.forward`.
  Template for the bit-accumulation readout loop (`for i, bit in enumerate(reversed(range(1, weight_precision)))`
  with `2 ** (bit - 1)` weighting). The embedding forward wraps this loop in a
  per-active-tile one-hot construction instead of consuming a dense `inputs` tensor.
- `synaptogen_ml/memristor_modules/memristor.py:273-290` — `PairedMemristorArrayV2`
  (used unchanged) and `:309-347` — `DacAdcPair` (DAC/ADC quantization, used unchanged).
- `synaptogen_ml/memristor_modules/config.py:8-20` — `CycleCorrectionSettings`
  (contract for the `correction_settings` arg).

### For `EmbeddingQuant` (the source type, no change)

- `recipe/model_pipelines/common/memristor_layers.py:221-275` — recipe `EmbeddingQuant`. Note
  `weight` shape `[V, D]`, and that it carries `weight_quantizer` /
  `weight_bit_prec` / `weight_quant_dtype` / `weight_quant_method` /
  `padding_idx` — the exact surface `init_from_embedding_quant` reads.
- `synaptogen_ml/quant_modules.py:225-263` — library `LinearQuant`, for comparison:
  same `weight_quantizer` plumbing, but `weight` shape `[out, in]`. This shape
  difference is the sole reason `init_from_embedding_quant` needs the extra transpose.

### For `_convert_linear` (File 3c)

This helper is the canonical `LinearQuant + ActivationQuantizer → TiledMemristorLinear`
conversion, factored out of the conformer FFN's `prep_quant`. It is not new logic —
it is the body of the reference below with the linear/quantizer pair passed in as args.

- `recipe/model_pipelines/common/assemblies/conformer/modules.py:137-165` —
  `ConformerPositionwiseFeedForwardQuant.prep_quant`, `linear_ff` half. This is the
  single-line reference implementation: `set_scale_and_zp` on weight+input quantizers →
  optional prune → build `TiledMemristorLinear` → `init_from_linear_quant` → swap in →
  input quantizer becomes `nn.Identity()`. `_convert_linear` is exactly this, returning
  the built module instead of assigning to `self.linear_ff`.
- `recipe/model_pipelines/common/assemblies/conformer/modules.py:168-192` — the `linear_out` half of
  the same `prep_quant` (second linear in the FFN; confirms the pattern repeats verbatim).
- `recipe/model_pipelines/common/memristor_layers.py:773-819` — `QuantizedMultiheadAttention.prep_quant`
  (`out_proj` and `qkv_proj`): same pattern outside the conformer FFN, showing it is
  not FFN-specific. Includes the `pruning_config` branch copied into `_convert_linear`.
- `recipe/model_pipelines/common/memristor_layers.py:1003-1042` — `LSTMQuant.prep_quant`: the loop-over-
  gates variant of the same conversion, included to show the pattern is reused across
  layer types.

### For `_convert_embedding` (File 3c)

No direct precedent (this is the new wiring). It is the embedding analogue of
`_convert_linear`: same `set_scale_and_zp` → (optional prune) → build → `init_from_*`
→ swap-in sequence, but with `TiledMemristorEmbedding.init_from_embedding_quant`
instead of `init_from_linear_quant`, and no input `ActivationQuantizer` (embeddings take
indices, not activations, so there is no `in_quant.set_scale_and_zp()` and no
`nn.Identity()` swap). The `pruning_config` branch IS included — `EmbeddingQuant`
exposes `pruning_config` and applies it in `forward` (see
`recipe/model_pipelines/common/memristor_layers.py:232,257,262-263`).

### Why `prep_quant` takes no arguments

- `recipe/model_pipelines/common/memrecog.py:31` — `model.prep_quant()` called in
  `MemristorModelConversionJob.run` after `load_state_dict`. This is why
  `converter_hardware_settings` / `correction_settings` / `num_cycles` must be
  stashed on `self` in `__init__` (File 3b) rather than passed into `prep_quant`.

---

## File 1 (NEW): `synaptogen_ml/memristor_modules/embedding.py`

> Mirrors `synaptogen_ml/memristor_modules/linear.py:126-358` (`TiledMemristorLinear`).
> See the References section above for the per-method mapping.

```python
__all__ = ["TiledMemristorEmbedding"]

import numpy as np
import torch
from torch import nn
from typing import Optional

from ..synaptogen import CellArrayCPU
from .memristor import DacAdcHardwareSettings, DacAdcPair, PairedMemristorArrayV2
from .config import CycleCorrectionSettings


class TiledMemristorEmbedding(nn.Module):
    """
    Memristor-array embedding table (address-decoded ROM).

    ``F.embedding(idx, W)`` with ``W`` of shape ``[V, D]`` is a row-select. The
    hardware analog programs the transposed table into tiled paired memristor
    crossbar arrays (``V`` input lines, ``D`` output lines) and, on forward,
    drives a single input line selected by ``idx`` (address decode) instead of
    a full ``V``-wide one-hot matmul. Only the input tile holding the active
    index is read out; all other tiles stay idle.

    Construction mirrors :class:`TiledMemristorLinear`: weights are bit-sliced
    into ``weight_precision - 1`` planes, each plane tiled into
    ``memristor_inputs x memristor_outputs`` paired (positive/negative) arrays,
    read out through a shared :class:`DacAdcPair`.

    There is no input activation quantizer: the active line is driven at DAC
    full scale (value 1.0), so ``input_factor = 1.0`` and
    ``output_factor = weight_quantizer.scale``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_precision: int,
        converter_hardware_settings: DacAdcHardwareSettings,
        memristor_inputs: int,
        memristor_outputs: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight_precision = weight_precision

        self.input_tiling = (num_embeddings + memristor_inputs - 1) // memristor_inputs
        self.output_tiling = (embedding_dim + memristor_outputs - 1) // memristor_outputs
        self.memristor_inputs = memristor_inputs
        self.memristor_outputs = memristor_outputs

        self.memristors = torch.nn.ModuleList(
            [
                PairedMemristorArrayV2(memristor_inputs, memristor_outputs)
                for _ in range(
                    (weight_precision - 1) * self.input_tiling * self.output_tiling
                )
            ]
        )
        self.converter = DacAdcPair(hardware_settings=converter_hardware_settings)
        self.input_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.output_factor = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.initialized = False

    def get_memristor_index(self, bit_level_index, input_index, output_index):
        return (
            bit_level_index * (self.input_tiling * self.output_tiling)
            + input_index * self.output_tiling
            + output_index
        )

    def init_from_embedding_quant(
        self,
        embedding_quant,
        num_cycles_init: int,
        correction_settings: Optional[CycleCorrectionSettings],
    ):
        """
        Program the paired memristor arrays from a trained ``EmbeddingQuant``.

        :param embedding_quant: object exposing ``weight`` ``[V, D]``,
            ``weight_quantizer`` (with ``set_scale_and_zp`` and ``scale``), and
            ``weight_bit_prec``. The recipe's ``EmbeddingQuant`` satisfies this.
        :param num_cycles_init: programming cycles (0 => single-shot write).
        :param correction_settings: optional write-verify cycle correction.
        """
        # EmbeddingQuant.weight is already [V, D] = [in, out] — the layout that
        # TiledMemristorLinear.init_from_linear_quant reaches *after* its inner
        # transpose. Do NOT transpose here.
        #
        # NB: capture `weight_scale` AFTER the weight_quantizer() call, not before.
        # prep_quant runs with the model still in train mode (memrecog.py:26-32
        # never calls .eval()), so WeightQuantizer.forward re-observes the
        # (post-pruning) weight and recomputes scale during the call below.
        # Reading scale before would desync output_factor from the scale actually
        # used to program the arrays. This mirrors init_from_linear_quant, which
        # reads linear_quant.weight_quantizer.scale at linear.py:185 and :313
        # (after the forward at :179) and does NOT call set_scale_and_zp itself
        # (the caller _convert_embedding does — same as _convert_linear does for
        # _convert_linear's LinearQuant).
        quant_weights = embedding_quant.weight_quantizer(embedding_quant.weight).detach()
        weight_scale = embedding_quant.weight_quantizer.scale

        weights_sign = torch.sign(quant_weights)
        quant_weights_scaled_abs = torch.round(
            torch.absolute(quant_weights / weight_scale)
        ).to(dtype=torch.int32)

        fill_input = self.input_tiling * self.memristor_inputs - quant_weights.size(0)
        fill_output = self.output_tiling * self.memristor_outputs - quant_weights.size(1)

        for i, bit in enumerate(reversed(range(0, self.weight_precision - 1))):
            quant_weights_scaled_bit = quant_weights_scaled_abs // (2 ** bit)
            quant_weights_scaled_abs = quant_weights_scaled_abs % (2 ** bit)
            quant_weights_scaled = quant_weights_scaled_bit * weights_sign
            quant_weights_scaled_pad = nn.functional.pad(
                quant_weights_scaled,
                pad=(0, fill_output, 0, fill_input),
            )

            for j in range(self.input_tiling):
                for k in range(self.output_tiling):
                    slice_ = quant_weights_scaled_pad[
                        j * self.memristor_inputs : (j + 1) * self.memristor_inputs,
                        k * self.memristor_outputs : (k + 1) * self.memristor_outputs,
                    ]
                    flat = torch.flatten(slice_).cpu()
                    positive_weights = torch.clamp(flat, 0, 1).numpy()
                    negative_weights = torch.abs(torch.clamp(flat, -1, 0)).numpy()

                    size = flat.shape[0]
                    positive_cells = CellArrayCPU(size)
                    negative_cells = CellArrayCPU(size)
                    if (
                        correction_settings is not None
                        and correction_settings.ideal_programming
                    ):
                        positive_cells.r = np.ones_like(positive_weights) - positive_weights
                        negative_cells.r = np.ones_like(negative_weights) - negative_weights
                    else:
                        for _ in range(num_cycles_init * 15):
                            positive_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
                            negative_cells.applyVoltage(np.random.uniform(-2.0, 2.0))
                        positive_cells.applyVoltage(2.0)
                        negative_cells.applyVoltage(2.0)
                        positive_cells.applyVoltage(positive_weights * -2.0)
                        negative_cells.applyVoltage(negative_weights * -2.0)

                        if correction_settings is not None:
                            for _ in range(correction_settings.num_cycles):
                                tensor = (
                                    np.ones_like(positive_weights)
                                    * correction_settings.test_input_value
                                )
                                pos = (
                                    positive_cells.I(tensor)
                                    * self.converter.hs.hardware_output_current_scaling
                                )
                                neg = (
                                    negative_cells.I(tensor)
                                    * self.converter.hs.hardware_output_current_scaling
                                )
                                pos_dev = np.abs(pos - positive_weights)
                                neg_dev = np.abs(neg - negative_weights)
                                pos_mask = pos_dev > correction_settings.relative_deviation
                                neg_mask = neg_dev > correction_settings.relative_deviation
                                positive_cells.applyVoltage(pos_mask * positive_weights * 2.0)
                                positive_cells.applyVoltage(pos_mask * positive_weights * -2.0)
                                positive_cells.applyVoltage(pos_mask * (1 - positive_weights) * -2.0)
                                positive_cells.applyVoltage(pos_mask * (1 - positive_weights) * 2.0)
                                negative_cells.applyVoltage(neg_mask * negative_weights * 2.0)
                                negative_cells.applyVoltage(neg_mask * negative_weights * -2.0)
                                negative_cells.applyVoltage(neg_mask * (1 - negative_weights) * -2.0)
                                negative_cells.applyVoltage(neg_mask * (1 - negative_weights) * 2.0)

                    index = self.get_memristor_index(i, j, k)
                    self.memristors[index].init_from_paired_cell_array_input_major(
                        positive_cells, negative_cells
                    )

        self.input_factor = torch.nn.Parameter(
            torch.tensor(1.0), requires_grad=False
        )
        self.output_factor = torch.nn.Parameter(
            torch.as_tensor(weight_scale).detach().clone(),
            requires_grad=False,
        )
        self.initialized = True

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        :param indices: integer tensor of arbitrary shape ``[...]``.
        :return: ``[..., embedding_dim]`` memristor-read embedding rows.
        """
        assert self.initialized, "TiledMemristorEmbedding not initialized"

        orig_shape = indices.shape
        flat_idx = indices.reshape(-1).to(dtype=torch.long)
        N = flat_idx.shape[0]
        out = torch.zeros(N, self.embedding_dim, device=indices.device)

        tile_of = flat_idx // self.memristor_inputs  # [N] which input tile
        pos_of = flat_idx % self.memristor_inputs    # [N] line within that tile

        for j in range(self.input_tiling):
            mask = tile_of == j
            if not torch.any(mask):
                continue
            rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
            G = rows.shape[0]
            pos = pos_of[rows]  # [G]

            local = torch.zeros(G, self.memristor_inputs, device=indices.device)
            local[torch.arange(G, device=indices.device), pos] = 1.0
            dac_in = self.converter.dac(local * self.input_factor)  # [G, mem_in]

            group_out = torch.zeros(G, self.embedding_dim, device=indices.device)
            for i, bit in enumerate(reversed(range(1, self.weight_precision))):
                start_index = self.get_memristor_index(i, j, 0)
                outputs = torch.concatenate(
                    [
                        self.converter.adc(self.memristors[start_index + k].forward(dac_in))
                        for k in range(self.output_tiling)
                    ],
                    dim=-1,
                )  # [G, pad_out]
                outputs = outputs[..., : self.embedding_dim]
                group_out = group_out + outputs * (2 ** (bit - 1))

            out[rows] = group_out * self.output_factor

        if self.padding_idx is not None:
            out[flat_idx == self.padding_idx] = 0.0

        return out.reshape(*orig_shape, self.embedding_dim)
```

---

## File 2 (EDIT): `synaptogen_ml/memristor_modules/__init__.py`

Add one import line. Full new content:

```python
from .conv import *
from .embedding import *
from .linear import *
from .memristor import *
from .util import *
```

---

## File 3 (EDIT): `ffnn_transducer_qat_encoder_pred/pytorch_modules.py`

> `_convert_linear` mirrors `recipe/model_pipelines/common/assemblies/conformer/modules.py:137-192`
> (`ConformerPositionwiseFeedForwardQuant.prep_quant`).
> `_convert_embedding` is its embedding analogue (no preexisting precedent; see
> References).

### 3a. Imports (top of file)

Add to the existing `synaptogen_ml` import block:

```python
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.memristor_modules.embedding import TiledMemristorEmbedding
```

(`DacAdcHardwareSettings` / `CycleCorrectionSettings` are already imported —
keep them, just add the `embedding` import.)

### 3b. `__init__` — stash hardware settings

`MemristorModelConversionJob.run` (`model_pipelines/common/memrecog.py:31`) calls
`model.prep_quant()` with no arguments, but the hardware settings live only in
`cfg`. Add at the end of `FFNNTransducerQATEncoderPredictionModel.__init__`,
after `self.joint_net = ...`:

```python
        self.converter_hardware_settings = cfg.converter_hardware_settings
        self.correction_settings = cfg.correction_settings
        self.num_cycles = cfg.num_cycles
```

### 3c. `prep_quant` + helpers

Insert after `__init__` (before `forward_encoder`):

```python
    def _convert_linear(self, lin: LinearQuant, in_quant: ActivationQuantizer) -> torch.nn.Module:
        try:
            from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
        except ModuleNotFoundError:
            from torch_memristor.memristor_modules import TiledMemristorLinear

        lin.weight_quantizer.set_scale_and_zp()
        in_quant.set_scale_and_zp()
        if lin.pruning_config is not None:
            with torch.no_grad():
                lin.weight.data = lin.pruning_config.apply(lin.weight.data, training=False)
        weight_prec = lin.weight_bit_prec if lin.weight_bit_prec != 1.5 else 2
        mem_lin = TiledMemristorLinear(
            in_features=lin.in_features,
            out_features=lin.out_features,
            weight_precision=weight_prec,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        mem_lin.init_from_linear_quant(
            activation_quant=in_quant,
            linear_quant=lin,
            num_cycles_init=self.num_cycles,
            correction_settings=self.correction_settings,
        )
        return mem_lin

    def _convert_embedding(self, emb: EmbeddingQuant) -> torch.nn.Module:
        emb.weight_quantizer.set_scale_and_zp()
        if emb.pruning_config is not None:
            with torch.no_grad():
                emb.weight.data = emb.pruning_config.apply(emb.weight.data, training=False)
        weight_prec = emb.weight_bit_prec if emb.weight_bit_prec != 1.5 else 2
        mem_emb = TiledMemristorEmbedding(
            num_embeddings=emb.num_embeddings,
            embedding_dim=emb.embedding_dim,
            weight_precision=weight_prec,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
            padding_idx=emb.padding_idx,
        )
        mem_emb.init_from_embedding_quant(
            embedding_quant=emb,
            num_cycles_init=self.num_cycles,
            correction_settings=self.correction_settings,
        )
        return mem_emb

    def prep_quant(self):
        self.conformer.prep_quant()

        self.token_embedding = self._convert_embedding(self.token_embedding)

        for base in range(0, len(self.prediction_net), 5):
            self.prediction_net[base + 2] = self._convert_linear(
                self.prediction_net[base + 2], self.prediction_net[base + 1]
            )
            self.prediction_net[base + 1] = torch.nn.Identity()
            self.prediction_net[base + 3].set_scale_and_zp()

        self.encoder_output[2] = self._convert_linear(self.encoder_output[2], self.encoder_output[1])
        self.encoder_output[1] = torch.nn.Identity()
        self.enc_output_in_q = torch.nn.Identity()
        self.enc_output_out_q.set_scale_and_zp()

        self.prediction_output[2] = self._convert_linear(
            self.prediction_output[2], self.prediction_output[1]
        )
        self.prediction_output[1] = torch.nn.Identity()
        self.prediction_output_in_q = torch.nn.Identity()
        self.prediction_output_out_q.set_scale_and_zp()
```

### Prediction-net layer layout note

`prediction_net` is built in `__init__` as a flat `nn.Sequential` of 5 entries
per layer, in this exact order (lines 113-147):

| index in block | module                              | action in `prep_quant`        |
|----------------|-------------------------------------|-------------------------------|
| `base + 0`     | `torch.nn.Dropout`                  | untouched                     |
| `base + 1`     | `ActivationQuantizer` (input)       | -> `nn.Identity()` (baked in) |
| `base + 2`     | `LinearQuant`                       | -> `TiledMemristorLinear`     |
| `base + 3`     | `ActivationQuantizer` (output)      | `set_scale_and_zp()` (stays)  |
| `base + 4`     | `cfg.pred_activation`               | untouched                     |

This matches the conformer FFN conversion at
`model_pipelines/common/assemblies/conformer/modules.py:137-192`.

---

## Verification

After editing, sanity-check in the base env:

```bash
cd /work/smt4/azim.javed/repositories/SynaptogenML
python -c "from synaptogen_ml.memristor_modules import TiledMemristorEmbedding; print(TiledMemristorEmbedding)"
```

There is no project-wide lint/test; per AGENTS.md this is a Sisyphus setup, not
a buildable project. Numerical equivalence of `TiledMemristorEmbedding` to
`F.embedding` (under `ideal_programming=True`, no noise) can be checked with a
tiny standalone script if desired before wiring it into the recipe.

## Out of scope (intentionally)

- `joint_net` is plain `nn.Linear` (not `LinearQuant`) -> not converted, same as
  the sibling `ffnn_transducer_qat_encoder` setup.
- The `mem_inited` variant of this model (if one exists) uses
  `TiledMemristorLinear` directly in `__init__`; an analogous
  `TiledMemristorEmbedding`-in-`__init__` path is a separate change.
- No changes to `EmbeddingQuant` itself.
