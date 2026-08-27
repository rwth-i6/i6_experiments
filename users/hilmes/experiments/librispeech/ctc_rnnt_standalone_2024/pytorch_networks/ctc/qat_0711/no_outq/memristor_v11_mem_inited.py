"""
no_outq variant of memristor_v11_mem_inited: identical model, distinct module name, with the
activation output quantizers (*_out_quant) replaced by nn.Identity -- like the input quantizers
already are in the _mem_inited modules.

Recognition only: the eval pipeline appends "_mem_inited" to the recognition network module, so
passing "ctc.qat_0711.no_outq.memristor_v11" resolves to this file, while the conversion and
prior jobs keep the plain v11/v10 modules and stay cached. The unmodified converted_model.pt is
loaded as-is; the out-quant observer keys become unexpected keys, which MiniReturnn tolerates
(load_state_dict with strict=False).

Reference variant: this reproduces the hand-written memristor_v11_no_outq_mem_inited.py, which is
kept only for the jobs that already ran with it.
"""

from ..memristor_v11_mem_inited import *  # noqa: F401,F403
from ..memristor_v11_mem_inited import Model as _BaseModel
from ._strip import strip_output_quantizers


class Model(_BaseModel):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__(model_config_dict, **kwargs)
        strip_output_quantizers(self)
