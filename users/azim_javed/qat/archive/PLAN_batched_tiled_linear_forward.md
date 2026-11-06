# Plan: Vectorize the tiling loop in `TiledMemristorLinear.forward`

## Status

- **Proposed** — not yet implemented.
- Targets `synaptogen_ml/memristor_modules/linear.py:321-358` (`TiledMemristorLinear.forward`).
- Preserves the existing eager (bit-exact) path and the `torch.compile`-fused fast inference path; only changes how the fast path is *called* for the tile loop.

## Context

### Symptom

A `ReturnnForwardJobV2` forward over 28,125 librispeech sequences with a 1.06B-param memristor-Conformer runs at a flat **~82 sec/step** even after the Inductor/Triton cache fix (which already brought it down from a flat ~139 sec/step). The remaining time is not compute — it is kernel-launch overhead.

### Root cause

`TiledMemristorLinear.forward` (`linear.py:321-358`) is a triple-nested Python loop:

```python
for i, bit in enumerate(reversed(range(1, self.weight_precision))):   # 7 iterations (weight_precision=8)
    for j in range(self.input_tiling):                                 # ~4-16
        outputs = torch.concatenate([
            self.converter.adc(self.memristors[start_index + k].forward(input_slice))
            for k in range(self.output_tiling)                         # ~4-16
        ], dim=-1)
```

Each leaf call goes through `PairedMemristorArrayV2.forward` → `pos.forward − neg.forward` → the compiled fused forward (`memristor.py:260`). The fused kernel optimizes *each* `MemristorArray.forward`, but the outer tiling loop still produces one small CUDA launch per (bit, input_tile, output_tile, polarity).

For a 512→2048 FF linear with `memristor_inputs=memristor_outputs=128`:

- `input_tiling = ceil(512/128) = 4`
- `output_tiling = ceil(2048/128) = 16`
- arrays per layer = `(weight_precision−1) × input_tiling × output_tiling = 7 × 4 × 16 = 448`
- calls per layer = `448 × 2` (pos/neg) = **896 fused-forward launches**

With ~80 memristor linears across the 12-block Conformer, that is **~65k tiny kernel launches per forward step**. At ~82 sec/step that is ~1.3 ms per launch — pure launch overhead, not compute.

### Why the conv layer is not a bottleneck

`MemristorConv1d.forward` (`conv.py:199-247`) already pre-broadcasts across output channels and uses only `(weight_precision−1)` arrays total — one `MemristorArray.forward` call per bit, no tiling loop. **Conv is already efficient; only `TiledMemristorLinear` needs work.**

### Key facts that enable the fix

Confirmed by reading the source:

1. **Polynomial coefficients are global, not per-array.** They come from `default_params.json` (`LLRS` / `HHRS`) and are loaded identically into every `CellArrayCPU` via `CellParams`. Every `MemristorArray.resistance_weighted_poly_low/high` holds the same `[P]` vector. Only `r` differs per array.
2. **`memristor_inputs = memristor_outputs = 128`** everywhere in the conformer recipe (`recipe/model_pipelines/common/assemblies/conformer/mem_inited/modules.py:65,75,...`).
3. **`DacAdcPair.dac` / `adc` are stateless** — they only use the shared `DacAdcHardwareSettings`, so they can be applied once to a batched output rather than per tile.
4. **`PairedMemristorArrayV2.forward = pos.forward(inp) − neg.forward(inp)`** (`memristor.py:289-290`) — the pos/neg split can be embedded as an extra dim of a batched `r`.
5. **`_get_compiled_fused_forward()`** (`memristor.py:45-54`) is a single module-level `torch.compile(_fused_memristor_forward, dynamic=True)`, shared across all `MemristorArray` instances. It accepts arbitrary leading batch dims. We can reuse it unchanged.

## Goal

Collapse the `weight_precision × input_tiling × output_tiling × polarity` launches in `TiledMemristorLinear.forward` to **`weight_precision − 1`** launches (one per bit), by batching all tiles of a bit into a single fused-forward call.

Expected launch-count reduction: **896 → 7 per FF linear**, i.e. **~100× fewer launches**. With ~80 memristor linears per forward, total launches drop from ~67k to ~560 per step. Step time should fall from ~82 s to low single-digit seconds, assuming the GPU is not then compute-bound.

## Contract to preserve

- **Eager path stays bit-exact** and unchanged. Keep the existing loop as `_forward_eager` and gate the new path on `is_fast_inference()`.
- **Use the same compiled fused forward** as `MemristorArray._forward_fast` — do not introduce a new compiled kernel. This keeps the dynamo cache shared and avoids per-instance recompilation (the exact pitfall `memristor.py:26-30` warns about).
- **`dac`/`adc` applied at the same points** with identical scales (they use the shared `DacAdcHardwareSettings`).
- **`init_from_linear_quant` is untouched.** It keeps writing per-array `r` into the `ModuleList` exactly as today; the batched-`r` buffer is built lazily on the first fast forward, so checkpoint loading and the conversion pipeline are unaffected.
- **Documented ~1e-6 numerical drift of the fast path is unchanged in nature.** See "Numerical-equivalence caveat" below for the one behavioral shift.

## Implementation

All edits are in `synaptogen_ml/memristor_modules/linear.py`. No changes to `memristor.py`, `conv.py`, `quant_modules.py`, or any recipe/config code.

### Step 1 — keep the old loop as `_forward_eager`

Rename the current `forward` body to `_forward_eager(self, inputs)` verbatim. This becomes the bit-exact reference path, used when `is_fast_inference()` is False and for numerics validation.

### Step 2 — add a lazy batched-`r` builder

Add a helper that materializes a single `[num_bits, 2, input_tiling, output_tiling, MI, MO]` buffer from the per-array `r` parameters. Build it on first fast forward and cache it on the module (`self._r_batched`). Invalidate on any `init_from_linear_quant` call (set `self._r_batched = None` there).

Memory: for FF (512→2048), `7 × 2 × 4 × 16 × 128 × 128 × 4 B ≈ 59 MB`. For the largest linears (qkv 512→1536, `it=4, ot=12`): ~44 MB. Total across all linears in a 12-block Conformer: well under 1 GB — acceptable given the job already sits at 15 GB RSS / 250 GB VMS.

```python
def _build_batched_r(self):
    nb = self.weight_precision - 1
    it, ot = self.input_tiling, self.output_tiling
    MI, MO = self.memristor_inputs, self.memristor_outputs
    dev = self.memristors[0].pos.r.device
    r = torch.empty(nb, 2, it, ot, MI, MO, device=dev, dtype=torch.float32)
    for i in range(nb):
        for j in range(it):
            for k in range(ot):
                idx = self.get_memristor_index(i, j, k)
                r[i, 0, j, k] = self.memristors[idx].pos.r
                r[i, 1, j, k] = self.memristors[idx].neg.r
    self._r_batched = r
```

This loop runs **once per linear**, at first forward — negligible.

### Step 3 — batched fast forward

New `forward` dispatches:

```python
def forward(self, inputs):
    assert self.initialized
    assert not self.output_factor == 1.0, ("Is the model properly initialized?", self.output_factor)
    if not is_fast_inference():
        return self._forward_eager(inputs)
    return self._forward_fast_batched(inputs)
```

`_forward_fast_batched` issues one fused call per bit:

```python
def _forward_fast_batched(self, inputs):
    nb = self.weight_precision - 1
    it, ot = self.input_tiling, self.output_tiling
    MI, MO = self.memristor_inputs, self.memristor_outputs

    if not hasattr(self, "_r_batched") or self._r_batched is None:
        self._build_batched_r()

    # dac once on the full input
    inp = self.converter.dac(inputs * self.input_factor)        # [...B, I]
    inp = F.pad(inp, (0, it * MI - inp.size(-1)))               # [...B, I_padded]
    inp = inp.reshape(*inp.shape[:-1], it, MI)                  # [...B, it, MI]

    # global poly coefficients (shared across all arrays in this layer)
    poly_low  = self.memristors[0].pos.resistance_weighted_poly_low
    poly_high = self.memristors[0].pos.resistance_weighted_poly_high

    # reuse the SAME compiled fused forward as MemristorArray._forward_fast
    fused = _get_compiled_fused_forward()

    out = torch.zeros(*inputs.shape[:-1], self.out_features, device=inp.device)
    for b in range(nb):                                          # 7 iterations
        rb = self._r_batched[b]                                 # [2, it, ot, MI, MO]

        # broadcast input across ot and flatten (2, it, ot) into one batch dim
        x = inp.unsqueeze(-2).expand(*inp.shape[:-1], ot, MI)    # [...B, it, ot, MI]
        x = x.reshape(*x.shape[:-3], 2 * it * ot, MI)           # [...B, 2*it*ot, MI]
        rb = rb.reshape(2 * it * ot, MI, MO)                    # [2*it*ot, MI, MO]

        # one noise draw per (polarity, it, ot), broadcast over [...B]
        noise = torch.randn(rb.shape, device=inp.device)        # [2*it*ot, MI, MO]
        noise = noise.expand(*inputs.shape[:-1], *noise.shape)

        result = fused(
            poly_low, poly_high, rb, x, noise,
            self.memristors[0].pos.kBT,
            self.memristors[0].pos.BW,
            self.memristors[0].pos.e,
            self.memristors[0].pos.noise_minimum_voltage,
        )                                                       # [...B, 2*it*ot, MO]

        result = result.reshape(*inputs.shape[:-1], 2, it, ot, MO)
        result = result[..., 0, :, :, :] - result[..., 1, :, :, :]   # paired subtract
        result = self.converter.adc(result)                     # adc once per bit
        result = result.reshape(*inputs.shape[:-1], it, ot * MO)
        result = result[..., : self.out_features]               # truncate output padding

        bit = (self.weight_precision - 1) - b                   # 7, 6, ..., 1
        out = out + result.sum(dim=-2) * (2 ** (bit - 1))      # sum over input tiles

    out = out * self.output_factor
    if self.bias is not None:
        out = out + self.bias
    return out
```

Notes:
- `poly_mul_horner` inside `_fused_memristor_forward` already broadcasts over arbitrary leading dims, so `[...B, 2*it*ot, MI]` works without modification.
- The `r`-broadcasting (`result_low * (1 − r) + result_high * r`) likewise broadcasts over the batched leading dims.
- Output padding truncation (`[: self.out_features]`) mirrors `linear.py:351`.
- Bit weighting `2 ** (bit − 1)` mirrors `linear.py:354`.

### Step 4 — invalidate the cache on re-init

In `init_from_linear_quant`, after `self.initialized = True`, add:

```python
self._r_batched = None
```

so a subsequent forward rebuilds from the new `r` values. Also initialize `self._r_batched = None` in `__init__`.

### Step 5 — (optional) numerics guard

Add a `SYN_FAST_VALIDATE` env check (same pattern as `SYN_FAST` / `SYN_NO_COMPILE` in `synaptogen_ml/__init__.py`). When set, run both `_forward_eager` and `_forward_fast_batched` on the first N forward calls and assert relative error below a tolerance:

```python
if os.environ.get("SYN_FAST_VALIDATE"):
    eager = self._forward_eager(inputs)
    fast = self._forward_fast_batched(inputs)
    rel = (fast - eager).abs().max() / eager.abs().max().clamp(min=1e-12)
    assert rel < 1e-4, f"batched fast path diverged: rel_err={rel}"
    return fast
```

Use this for one short run to confirm equivalence before trusting it on the full 28k-seq forward. Disabled by default.

## Numerical-equivalence caveat

The noise draw order changes. Today each tile does its own `torch.randn([MI, MO])` inside its fused-forward call. The batched path does a single `torch.randn([2*it*ot, MI, MO])` per bit. Per-tile noise is **statistically identical** (same distribution, same shape per tile), but the specific RNG sequence assigned to each (bit, polarity, input_tile, output_tile) will differ from the un-batched fast path.

This is consistent with the fast path's existing contract — `synaptogen_ml/__init__.py:5-10` already documents ~1e-6 drift from Horner/fusion and states the path is **not bit-exact** vs eager. The eager path itself is unaffected by this change (it stays as `_forward_eager`).

If any downstream test diffs fast-path output against a recorded reference down to the float, it will move. If such a pinned reference exists, regenerate it after this change.

## Risks and open questions

1. **Peak memory.** The batched fused call produces a `[..., 2*it*ot, MI, MO]` intermediate. For FF with `B` sequences and `T'` time steps and `2*it*ot = 128`, that's `B × T' × 128 × 128 × 128 × 4 B`. Worth measuring on the real batch size (`batch_size=3200000` frames) before declaring victory — if it blows the 24 GB A10/3090, fall back to sub-batching over `ot` (e.g. process 4 output tiles at a time: 7 × ceil(16/4) = 28 launches instead of 7, still a 32× reduction).
2. **Noise determinism.** If reproducible stochasticity across runs matters, fix the RNG seed before the batched draw (same as today) and document that noise assignments have been repartitioned across tiles.
3. **`dynamic=True` and shape specialization.** The batched call has one more leading dim rank than before. Inductor with `dynamic=True` should handle it, but confirm the dynamo cache isn't recompiling per layer. `TORCH_COMPILE_DEBUG=1` on a short run will show any `recompiling` messages.
4. **Other `TiledMemristorLinear` callers.** Confirm no caller depends on the loop's intermediate structure (e.g. hooks, grad-capture). None seen in the conformer recipe, but a repo-wide grep for `TiledMemristorLinear` is worth doing before merging.
5. **MHSA `qkv_proj` (512→1536, `ot=12`).** Same path applies; just confirm the output-padding truncation handles `out_features` not divisible by `MO=128` correctly (it does — `[: self.out_features]` slices the `ot*MO` axis).

## Validation plan

1. **Unit test:** create a tiny `TiledMemristorLinear` (e.g. `in_features=10, out_features=20, memristor_inputs=4, memristor_outputs=4, weight_precision=4`), call `init_from_linear_quant` with a dummy `LinearQuant`+`ActivationQuantizer`, run `forward` in both eager and batched-fast modes, assert `|fast − eager| / |eager| < 1e-4`.
2. **Integration smoke test:** load the real `memristor_converted_model.pt`, set `SYN_FAST=1` and `SYN_FAST_VALIDATE=1`, run one forward step on a few sequences, confirm the assertion passes.
3. **Speed test:** rerun the `ReturnnForwardJobV2` forward with the patched `synaptogen_ml`, confirm step 0 is one-time-compile-slow and step 1+ drops to single-digit seconds. Compare `prior.txt` against a pre-patch run — they should agree to within ~1e-4 relative on the averaged log-probs (the memo's point of the forward).
4. **Memory watch:** watch `nvidia-smi` during the first few steps to confirm the batched intermediates fit in 24 GB; if not, apply the `ot`-sub-batching fallback from Risk 1.

## Files to change

- `synaptogen_ml/memristor_modules/linear.py` — rename old `forward` to `_forward_eager`, add `_build_batched_r`, add `_forward_fast_batched`, change `forward` to dispatch, set `self._r_batched = None` in `__init__` and at the end of `init_from_linear_quant`.

No other files change.

## Pointers

- Slow job log (post cache-fix, the ~82 s/step baseline this plan targets): `i6_core/returnn/forward/ReturnnForwardJobV2.IUYzAYhfiMSk/log.run.1`
- `TiledMemristorLinear.forward` (the loop to vectorize): `synaptogen_ml/memristor_modules/linear.py:321`
- `MemristorArray.forward` / `_forward_fast` (the per-array fused kernel we reuse): `synaptogen_ml/memristor_modules/memristor.py:198`, `:217`
- `_get_compiled_fused_forward` (the shared `torch.compile`'d kernel): `synaptogen_ml/memristor_modules/memristor.py:45`
- `_fused_memristor_forward` (the function being compiled; see its broadcasting contract): `synaptogen_ml/memristor_modules/memristor.py:17`
- Conformer recipe wiring (tile sizes 128×128): `recipe/model_pipelines/common/assemblies/conformer/mem_inited/modules.py:58-75`
- Fast-inference env switch (`SYN_FAST`): `synaptogen_ml/__init__.py:22`