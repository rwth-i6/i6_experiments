"""
Energy-measure variant of memristor_v16_dynmic_prec_mem_inited: identical model, distinct
module name.

The eval pipeline appends "_mem_inited" to the recognition network module, so passing
"...claude.memristor_v16_dynmic_prec_energy" as recognition module resolves to this file.
The module name is part of the job hash, which triggers the recognition rerun; conversion
and prior jobs keep the plain v16 modules and stay cached. The instrumentation itself is
attached by the energy decoder (ctc.decoder.rasr_ctc_v1_batched_energy) via
claude/energy_measure.py.
"""

from .memristor_v16_dynmic_prec_mem_inited import Model as _BaseModel
from .memristor_v16_dynmic_prec_mem_inited import (  # noqa: F401
    train_step,
    prior_init_hook,
    prior_step,
    prior_finish_hook,
    mem_init_hook,
    mem_step,
    mem_finish_hook,
    compute_stats_init_hook,
    compute_stats_step,
    compute_stats_finish_hook,
)


class Model(_BaseModel):
    is_energy_measure_variant = True
