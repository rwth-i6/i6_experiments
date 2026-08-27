"""
RASR CTC decoder that measures memristor currents / device usage WHILE running
SynaptogenML's fast inference path (summary detail only).

Contrast rasr_ctc_v1_batched_energy: that one needs the eager compute path (its
per-cell capture hook lives inside the energy pin's MemristorArray.forward) plus
the dedicated "_energy" model variant and the energy pin. This variant instead
wraps each array's forward from outside (see
qat_0711/claude/energy_measure.FastEnergyCollector), so it works with the
STANDARD recognition network module on the standard v3+ pins -- conversion and
prior jobs are shared with the plain recognition, and the forward runs at fast
speed. Column-level statistics are exact; the per-cell supply sums are analytic
(noise-free) approximations, validated against the eager arm in the
*_energy_ab_* runs.

Measurement options arrive as decoder_args["energy"] (plain dict for
EnergyMeasureConfig, detail must be "summary"); the job must declare
energy_report.pkl / energy_summary.json as additional output files.
"""

from . import rasr_ctc_v1_batched as _base
from .rasr_ctc_v1_batched import DecoderConfig, ExtraConfig  # noqa: F401

ENERGY_REPORT_PICKLE = "energy_report.pkl"
ENERGY_REPORT_JSON = "energy_summary.json"


def forward_init_hook(run_ctx, **kwargs):
    try:
        import torch_memristor as _synaptogen_pkg
    except ModuleNotFoundError:
        import synaptogen_ml as _synaptogen_pkg
    assert hasattr(_synaptogen_pkg, "set_fast_inference"), (
        "fast_energy decoder needs a SynaptogenML pin with set_fast_inference "
        "(v3+); route via import_memristor='new_v3' or newer"
    )
    _synaptogen_pkg.set_fast_inference(True)

    _base.forward_init_hook(run_ctx, **kwargs)

    from ..qat_0711.claude.energy_measure import EnergyMeasureConfig, attach_fast

    model = run_ctx.engine._model
    # all measurement options must arrive explicitly via the hashed decoder args;
    # EnergyMeasureConfig has no defaults, so missing keys fail loudly here
    energy_config = EnergyMeasureConfig(**kwargs["energy"])
    run_ctx.energy_collector = attach_fast(model, energy_config)
    print(
        f"[energy_measure] fast collector attached to {len(run_ctx.energy_collector.infos)} "
        f"memristor arrays, config: {energy_config}"
    )


def forward_step(*, model, data, run_ctx, **kwargs):
    _base.forward_step(model=model, data=data, run_ctx=run_ctx, **kwargs)
    run_ctx.energy_collector.finish_batch()


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.energy_collector.write_report(ENERGY_REPORT_PICKLE, ENERGY_REPORT_JSON)
    _base.forward_finish_hook(run_ctx, **kwargs)
