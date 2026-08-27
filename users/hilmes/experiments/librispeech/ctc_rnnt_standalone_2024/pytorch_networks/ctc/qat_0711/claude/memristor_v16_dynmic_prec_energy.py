"""
Energy-measure variant of memristor_v16_dynmic_prec: identical model, distinct module name.

The network-module name is part of every job hash, so recognitions configured with this
module rerun and are labelled as energy measurements, while conversion/prior jobs keep the
plain v16 module and stay cached. The actual instrumentation is attached by the energy
decoder (ctc.decoder.rasr_ctc_v1_batched_energy) via claude/energy_measure.py.
"""

from .memristor_v16_dynmic_prec import Model as _BaseModel
from .memristor_v16_dynmic_prec import (  # noqa: F401
    train_step,
    prior_init_hook,
    prior_step,
    prior_finish_hook,
    mem_init_hook,
    mem_step,
    mem_finish_hook,
    prune_init_hook,
    prune_step,
    prune_finish_hook,
)


class Model(_BaseModel):
    is_energy_measure_variant = True
