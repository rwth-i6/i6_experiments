"""
RASR CTC decoder variant that additionally measures memristor currents / device usage.

Delegates all search behavior to rasr_ctc_v1_batched and only attaches the energy
instrumentation (claude/energy_measure.py) around it:
  init:   original init (incl. prep_quant), then attach the collector to the model
  step:   original step (one encoder forward + search), then fold the per-batch peaks
  finish: write energy_report.pkl + energy_summary.json into the job dir, then original

Measurement options arrive as decoder_args["energy"] (plain dict for
EnergyMeasureConfig); the "config" entry stays the untouched rasr_ctc_v1_batched
DecoderConfig. The job must declare energy_report.pkl / energy_summary.json as
additional output files.
"""

from . import rasr_ctc_v1_batched as _base
from .rasr_ctc_v1_batched import DecoderConfig, ExtraConfig  # noqa: F401

ENERGY_REPORT_PICKLE = "energy_report.pkl"
ENERGY_REPORT_JSON = "energy_summary.json"


def forward_init_hook(run_ctx, **kwargs):
    _base.forward_init_hook(run_ctx, **kwargs)

    from ..qat_0711.claude.energy_measure import EnergyMeasureConfig, attach

    model = run_ctx.engine._model
    assert getattr(model, "is_energy_measure_variant", False), (
        "energy decoder expects an energy model variant (e.g. "
        "ctc.qat_0711.claude.memristor_v16_dynmic_prec_energy), got "
        f"{type(model).__module__}.{type(model).__name__}"
    )
    # all measurement options must arrive explicitly via the hashed decoder args;
    # EnergyMeasureConfig has no defaults, so missing keys fail loudly here
    energy_config = EnergyMeasureConfig(**kwargs["energy"])
    run_ctx.energy_collector = attach(model, energy_config)
    print(
        f"[energy_measure] attached to {len(run_ctx.energy_collector.infos)} memristor arrays, "
        f"config: {energy_config}"
    )


def forward_step(*, model, data, run_ctx, **kwargs):
    _base.forward_step(model=model, data=data, run_ctx=run_ctx, **kwargs)
    run_ctx.energy_collector.finish_batch()


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.energy_collector.write_report(ENERGY_REPORT_PICKLE, ENERGY_REPORT_JSON)
    _base.forward_finish_hook(run_ctx, **kwargs)
