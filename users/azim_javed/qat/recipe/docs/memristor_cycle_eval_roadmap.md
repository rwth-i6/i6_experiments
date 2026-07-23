# Memristor Cycle Evaluation — Implementation Roadmap

This document describes how to replicate the `run_memristor_cycle_eval` flow
(originally in
`/u/hilmes/experiments/asr_2023/recipe/i6_experiments/users/hilmes/experiments/librispeech/ctc_rnnt_standalone_2024/experiments/ctc_bpe/memristor.py`)
in the QAT pipeline at `training/qat/recipe/`.

The goal is to take a trained QAT model and simulate its behaviour when
deployed on a memristor crossbar for different numbers of programming cycles,
sweeping over DAC/ADC hardware configurations, LM scales and prior scales.

---

## Architecture overview (old setup)

```
memristor.py: run_memristor_cycle_eval()
    │
    ├── eval_model(split_mem_init=True, import_memristor=True)
    │       │
    │       ├── prepare_asr_model()
    │       │       └── prepare_memristor()  →  ReturnnForwardJobV2
    │       │              runs mem_init_hook → model.prep_quant()
    │       │              → replaces LinearQuant with TiledMemristorLinear
    │       │              → saves converted_model.pt    ◄── CONVERSION
    │       │
    │       └── tune_and_evaluate_helper()
    │               └── search()
    │                       ├── serialize_forward(import_memristor=True)
    │                       │       → ExternalImport(TORCH_MEMRISTOR_PATH)
    │                       └── search_single()  →  ReturnnForwardJobV2
    │                              loads converted_model.pt
    │                              forward pass: TiledMemristorLinear.forward()
    │                              → DAC → conductance → current sum → ADC
    │                              → posteriors → RASR decoder → WER
    │
    └── generate_report()  →  WER vs. cycle count
```

---

## Prerequisites — already in your codebase

The hardest integration — **the actual memristor hardware simulation layers** —
is already wired into the conformer modules. Here is what exists today:

| Component | File | Status |
|-----------|------|--------|
| `TiledMemristorLinear` (synaptogen\_ml) | imported in `memristor_layers.py` and `assemblies/conformer/modules.py` | ✅ |
| `MemristorConv1d` | imported in `assemblies/conformer/modules.py` | ✅ |
| `DacAdcHardwareSettings` | imported via `synaptogen_ml.memristor_modules` | ✅ |
| `CycleCorrectionSettings` | imported in `assemblies/conformer/config.py` | ✅ |
| `prep_quant()` on FF block | `ConformerPositionwiseFeedForwardQuantV4.prep_quant()` in `assemblies/conformer/modules.py:132` | ✅ |
| `prep_quant()` on MHSA block | `QuantizedMultiheadAttention.prep_quant()` in `memristor_layers.py:761` | ✅ |
| `prep_quant()` on Conv block | `ConformerConvolutionQuant.prep_quant()` in `assemblies/conformer/modules.py:387` | ✅ |
| `prep_quant()` on ConformerBlock | calls sub-module `prep_quant()` in `assemblies/conformer/modules.py:247` | ✅ |
| `prep_quant()` on ConformerEncoder | calls sub-module `prep_quant()` in `assemblies/conformer/modules.py:558` | ✅ |
| `WeightPruningConfig` and pruning logic | `assemblies/conformer/config.py` | ✅ |

**What is missing**: the job orchestration — the conversion job, the per-cycle
recognition, and the loop that ties them together.

---

## Phase 1 — Checkpoint conversion job (`prepare_memristor`)

### Goal

Create a `ReturnnForwardJobV2` that loads a trained QAT checkpoint, calls
`model.prep_quant()` (which replaces every `LinearQuant` / `Conv1dQuant`
with `TiledMemristorLinear` / `MemristorConv1d`), and saves
`converted_model.pt`.

### What the old setup does

```python
# pipeline.py :: prepare_memristor()
returnn_config = get_mem_init_config(
    network_module=..., config=...,
    net_args=..., import_memristor=True,
)
search_job = ReturnnForwardJobV2(
    model_checkpoint=checkpoint,
    returnn_config=returnn_config,
    output_files=["converted_model.pt"],
    device="cpu",
    cpu_rqmt=32,
    time_rqmt=72,
)
```

The `get_mem_init_config` creates a `ReturnnConfig` whose `python_epilog`
serializes a `forward_step` named `"mem"`. That step (defined in
`memristor_v10_mem_inited.py`) is:

```python
def mem_init_hook(run_ctx, **kwargs):
    run_ctx.engine._model.prep_quant()
    torch.save({"model": run_ctx.engine._model.state_dict(), ...}, "converted_model.pt")

def mem_step(*, model, data, run_ctx, **kwargs):
    pass   # no actual forward — only conversion
```

### What you need to build

#### 1a. Forward hooks for the conversion job

Create a new file, e.g. `model_pipelines/qat_ctc/memristor_init.py`:

```python
def mem_init_hook(run_ctx, **kwargs):
    run_ctx.engine._model.prep_quant()
    torch.save(
        {"model": run_ctx.engine._model.state_dict(), "epoch": run_ctx.epoch, "step": run_ctx.engine._train_step},
        "converted_model.pt",
    )

def mem_step(*, model, data, run_ctx, **kwargs):
    pass
```

#### 1b. `prepare_memristor()` job factory

Create a function (e.g. in `model_pipelines/common/memristor_eval.py` or
alongside `prior.py`) that:

1. Builds a `ReturnnConfig` with:
   - `backend = "torch"`
   - `batch_size` and `forward` dataset (the CV set is sufficient)
   - `python_epilog` containing:
     - `get_model_serializers(model_class, model_config)` — same as training
     - `ExternalImport(synaptogen_ml_path)` — so `TiledMemristorLinear` is importable at runtime
     - `Import` of `mem_init_hook` → `forward_init_hook`
     - `Import` of `mem_step` → `forward_step`
2. Creates a `ReturnnForwardJobV2` with:
   - `output_files=["converted_model.pt"]`
   - `device="cpu"` (conversion is compute-heavy but needs no GPU)
   - `cpu_rqmt=32`, `time_rqmt=72`, `mem_rqmt=24`
3. Returns `job.out_files["converted_model.pt"]`

**Signature**:

```python
def prepare_memristor(
    descriptor: str,
    model_class: Type[ModuleType],
    model_config: ModelConfiguration,
    checkpoint: PtCheckpoint,
    data_config: DataConfig,
) -> tk.Path:
    ...
```

#### 1c. Add `synaptogen_ml` path to `tools.py`

The old setup has `TORCH_MEMRISTOR_PATH` which is a git clone or local path
pointing to the `synaptogen_ml` package. You need the same:

```python
synaptogen_ml_root = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/SynaptogenML",
    checkout_folder_name="synaptogen_ml",
).out_repository.copy()
synaptogen_ml_root.hash_overwrite = "SYNAPTOGEN_ML_ROOT"
```

Or point to a local checkout.

---

## Phase 2 — Memristor recognition config and model

### Goal

The recognition job needs to instantiate the model with **recognition-time**
DAC/ADC settings and `num_cycles` (which may differ from training values).
The training settings are stored inside the `prep_quant()`-converted layers,
but the recog job's model config controls the initialisation of new
`TiledMemristorLinear` instances.

### What the old setup does

The old setup builds a fresh `model_config_recog` for each cycle:

```python
model_config_recog = copy.deepcopy(model_config)
model_config_recog.converter_hardware_settings = recog_dac_settings   # e.g. 4-bit ADC
model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings
model_config_recog.num_cycles = num_cycles
```

Then passes `model_config_recog` into the recognition network module via
`net_args`. Inside the recog network's `Model.__init__`, the config values
flow into each sub-module's `converter_hardware_settings` field, which
`prep_quant()` reads when constructing the `TiledMemristorLinear`.

### What you need to build

#### 2a. Memristor recognition config

Modify `pytorch_modules.py` (or create a sub‑class) to accept the
recognition‑time hardware settings:

```python
@dataclass
class QATConformerCTCMemristorRecogConfig(QATConformerCTCConfig):
    converter_hardware_settings: Optional[DacAdcHardwareSettings] = None
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None
    num_cycles: int = 0
    correction_settings: Optional[CycleCorrectionSettings] = None
    # prior / blank fields inherited from QATConformerCTCRecogConfig or added here
```

#### 2b. Override the sub‑module configs

The simplest approach: after constructing the `QATConformerCTCConfig`, walk
through the nested config objects and override their hardware‑settings fields
before the model is instantiated. Alternatively, add a helper that builds the
full config tree with the recognition‑time values baked in.

A helper in `assemblies/conformer/config.py`:

```python
def override_hardware_settings(
    cfg: ConformerEncoderQuantV1Config,
    converter_hw: DacAdcHardwareSettings,
    pos_enc_hw: DacAdcHardwareSettings,
    num_cycles: int,
    correction: Optional[CycleCorrectionSettings],
) -> ConformerEncoderQuantV1Config:
    """Recursively overwrite the converter/pos_enc/correction fields in every
    sub‑config of a ConformerEncoderQuantV1Config."""
    ...
```

#### 2c. Serialize synaptogen\_ml into the recog job

In the old setup, `serialize_forward(import_memristor=True)` adds an
`ExternalImport(TORCH_MEMRISTOR_PATH)` to the serializer collection. You
need a similar variant of `get_model_serializers()` or a dedicated factory:

```python
def get_memristor_model_serializers(
    model_class: Type[ModuleType],
    model_config: ModelConfiguration,
) -> Collection:
    base = get_model_serializers(model_class, model_config)
    base.serializer_objects.insert(
        1, ExternalImport(synaptogen_ml_root)
    )
    return base
```

---

## Phase 3 — Cycle evaluation orchestrator

### Goal

`run_memristor_cycle_eval()` — the top‑level function that:

1. Computes the prior once (shared across all cycles)
2. Runs the checkpoint conversion once (`prepare_memristor`)
3. For each cycle `1 .. max_runs`:
   - Builds a recognition config with `num_cycles=N` and the
     recognition‑time DAC/ADC settings
   - Launches RASR recognition (via `recog_rasr_offline`) across the
     sweep of LM scales and prior scales
   - Collects WERs
4. Generates a report (WER vs. cycle count)

### What the old setup does

```python
def run_memristor_cycle_eval(train_job, train_data, train_config, model_config, ...):
    # Prior (shared, uses training DAC settings)
    prior_args = {"config": train_config, "network_module": prior_network_module, ...}
    
    for num_cycles in range(1, max_runs + 1):
        # Recog config with cycle-specific settings
        model_config_recog = copy.deepcopy(model_config)
        model_config_recog.converter_hardware_settings = recog_dac_settings
        model_config_recog.num_cycles = num_cycles
        
        # Recognition
        res_fixed = eval_model(
            training_name=...,
            train_job=train_job,
            train_args=train_args_recog,
            decoder_config=rasr_config,
            prior_args=prior_args,
            ...
        )
    
    # Report
    generate_report(results=res_fixed, exp_name=report_name)
```

### What you need to build

Create `experiments/librispeech/recognition/memristor_cycle_eval.py`:

```python
from dataclasses import asdict
from itertools import product
from typing import Optional

from sisyphus import tk

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech.bpe import vocab_to_bpe_size
from ....model_pipelines.common.memristor_eval import prepare_memristor
from ....model_pipelines.common.recog import recog_rasr_offline
from ....model_pipelines.common.serializers import get_memristor_model_serializers
from ....model_pipelines.common.train import TrainedModel
from ....model_pipelines.qat_ctc.prior import compute_priors
from ....model_pipelines.qat_ctc.pytorch_modules import QATConformerCTCConfig, QATConformerCTCRecogModel
from .common import BaseRecogVariant, run_single_bpe_variant


def run_memristor_cycle_eval(
    model: TrainedModel[QATConformerCTCConfig],
    recog_dac_settings: DacAdcHardwareSettings,
    posenc_dac_settings: DacAdcHardwareSettings,
    prior_scales: list[float],
    lm_scales: list[float],
    max_runs: int,
    corpora: list[str],
    bpe_size: int = 128,
    epoch: Optional[int] = None,
) -> dict:
    """
    Simulate memristor deployment for 1..max_runs programming cycles.

    :param model: trained QAT model (unconverted checkpoint)
    :param recog_dac_settings: DAC/ADC settings for recognition (e.g. 4-bit ADC)
    :param posenc_dac_settings: DAC/ADC settings for positional encoding converter
    :param prior_scales: CTC prior scales to sweep
    :param lm_scales: LM scales to sweep
    :param max_runs: maximum number of memristor cycles to simulate
    :param corpora: list of corpus names (e.g. ["dev-clean", "dev-other"])
    :param bpe_size: BPE vocabulary size
    :param epoch: which training epoch to use (defaults to best/last)
    :return: dict mapping cycle number → list of RecogResult
    """
    checkpoint = model.get_checkpoint(epoch)
    results_by_cycle = {}

    # ── Phase 3a: Prior (shared, uses training DAC settings) ──────────
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model.model_config,
        checkpoint=checkpoint,
    )

    # ── Phase 3b: Checkpoint conversion (shared) ──────────────────────
    from ....model_pipelines.qat_ctc.pytorch_modules import QATConformerCTCModel

    cv_data = librispeech_datasets.get_default_bpe_cv_data(bpe_size=bpe_size)
    converted_checkpoint = prepare_memristor(
        descriptor=model.descriptor + "/memristor_init",
        model_class=QATConformerCTCModel,
        model_config=model.model_config,
        checkpoint=checkpoint,
        data_config=cv_data,
    )

    # ── Phase 3c: Cycle loop ───────────────────────────────────────────
    for num_cycles in range(1, max_runs + 1):
        # Build recognition config with cycle-specific hardware settings
        memristor_config = _build_memristor_recog_config(
            base_config=model.model_config,
            recog_dac_settings=recog_dac_settings,
            posenc_dac_settings=posenc_dac_settings,
            num_cycles=num_cycles,
            prior_file=prior_file,
        )

        # Serializers include synaptogen_ml
        serializers = get_memristor_model_serializers(
            QATConformerCTCRecogModel, memristor_config
        )

        # Sweep (lm_scale, prior_scale)
        for lm_scale, prior_scale in product(lm_scales, prior_scales):
            variant = CTCRecogVariant(
                descriptor=f"cycle{num_cycles}_lm{lm_scale}_p{prior_scale}",
                search_algorithm_params=...,
                prior_scale=prior_scale,
            )
            recog_result = run_single_bpe_variant(
                model_descriptor=f"{model.descriptor}/memristor/cycle_{num_cycles}",
                checkpoint=converted_checkpoint,
                encoder_serializers=serializers,
                ...
            )
            results_by_cycle.setdefault(num_cycles, []).append(recog_result)

    # ── Phase 3d: Report ──────────────────────────────────────────────
    _generate_cycle_report(model.descriptor, results_by_cycle)

    return results_by_cycle


def _build_memristor_recog_config(...) -> QATConformerCTCConfig:
    """Deep‑copy the training config and override hardware settings."""
    ...
```

---

## Phase 4 — Wire into the experiment entry point

### Goal

Add a `run_memristor()` function to `experiments/librispeech/__init__.py`
that trains a QAT model and then runs the cycle evaluation.

### Structure

```python
def run_memristor(filename):
    # 1. Train (or reuse an existing TrainedModel)
    model = training.qat_ctc_bpe_param_sync.run(
        descriptor="qat_ctc_bpe_memristor",
        qat_args=dict(weight_bit_prec=4, activation_bit_prec=8, ...),
    )

    # 2. Sweep over ADC/DAC configurations
    for prec, range_bits in [(4, 8), (8, 8)]:
        recog_dac = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=prec,
            output_range_bits=range_bits,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
        posenc_dac = DacAdcHardwareSettings(  # 1‑bit pos enc
            input_bits=8, output_precision_bits=1, output_range_bits=7,
            hardware_input_vmax=0.6, hardware_output_current_scaling=8020.0,
        )

        results = memristor_cycle_eval.run_memristor_cycle_eval(
            model=model,
            recog_dac_settings=recog_dac,
            posenc_dac_settings=posenc_dac,
            prior_scales=[0.3, 0.5],
            lm_scales=[0.6, 0.8, 1.0],
            max_runs=10,
            corpora=["dev-other"],
        )
```

### Report generation

Add a report template (in `model_pipelines/common/report.py` or alongside)
that produces a table/graph of WER vs. cycle count.

---

## Phase 5 — Optional enhancements (later)

| Feature | Description | Old‑setup reference |
|---------|-------------|---------------------|
| **Greedy decoding** | No LM, no prior, single‑pass greedy CTC decoder | `greedy_bpe_ctc_quant_v1` |
| **Multi‑scale sweep** | One forward pass per cycle, then apply all (lm, prior) combinations on the same posteriors | `rasr_ctc_v1_batched_multi` + `search_multi` |
| **Energy measurement** | Instrument the forward pass to report estimated energy per layer | `claude.energy_measure.EnergyMeasureConfig`, `import_memristor="energy"` |
| **Weight pruning** | Zero out small weights before/during training; `WeightPruningConfig` already wired | `ThresholdPruningConfig`, `PercentilePruningConfig` |
| **Ideal (noise‑free) baseline** | `CycleCorrectionSettings(ideal_programming=True)` | `model_config_ideal.correction_settings = ideal` |
| **Mixed precision per layer** | Different `weight_bit_prec` for FF / MHSA / Conv blocks | `memristor_v14` with per‑layer precision |

---

## File creation / modification summary

| File | Action | Purpose |
|------|--------|---------|
| `model_pipelines/qat_ctc/memristor_init.py` | **Create** | `mem_init_hook`, `mem_step` for the conversion forward job |
| `model_pipelines/common/memristor_eval.py` | **Create** | `prepare_memristor()` — the conversion‑job factory |
| `model_pipelines/qat_ctc/pytorch_modules.py` | **Modify** | Add `QATConformerCTCMemristorRecogConfig` (or extend `QATConformerCTCRecogConfig`) |
| `model_pipelines/common/serializers.py` | **Modify** | Add `get_memristor_model_serializers()` that injects `ExternalImport(synaptogen_ml)` |
| `tools.py` | **Modify** | Add `synaptogen_ml_root` path (git clone or local) |
| `assemblies/conformer/config.py` | **Modify** | (Optional) add `override_hardware_settings()` helper |
| `experiments/librispeech/recognition/memristor_cycle_eval.py` | **Create** | `run_memristor_cycle_eval()` orchestrator |
| `experiments/librispeech/__init__.py` | **Modify** | Add `run_memristor()` entry point |
| `model_pipelines/common/report.py` | **Modify** | (Optional) add memristor‑specific report template |

---

## Key differences from the old setup

| Aspect | Old setup (`memristor.py`) | Your setup |
|--------|---------------------------|------------|
| **Scheduler** | Sisyphus + custom `eval_model` helper | Sisyphus + `recog_rasr_offline` |
| **Recognition** | RASR via `ReturnnForwardJobV2` with integrated RASR decoder | RASR via `ReturnnForwardJobV2` with `OfflineSearchCallback` (already in `recog.py`) |
| **Prior** | Computed by `prepare_asr_model` → `compute_prior` | Already have `compute_priors` in `qat_ctc/prior.py` |
| **Conversion job** | `prepare_memristor` in `pipeline.py` | **Needs to be built** (Phase 1) |
| **synaptogen\_ml import** | `ExternalImport(TORCH_MEMRISTOR_PATH)` in `serialize_forward` | **Needs to be built** (Phase 2c) |
| **Model serialization** | Custom `serialize_forward()` with `import_memristor` flag | `get_model_serializers()` — needs `synaptogen_ml` variant |
| **`prep_quant`** | Already in modules | Already in modules ✅ |
| **DAC/ADC overrides at recog time** | Deep‑copy config + override fields | **Needs to be built** (Phase 2a/2b) |
