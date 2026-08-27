"""
no_outq variant of memristor_v15_mem_inited: identical model, distinct module name, with the
activation output quantizers (*_out_quant) replaced by nn.Identity -- like the input quantizers
already are in the _mem_inited modules.

Recognition only: the eval pipeline appends "_mem_inited" to the recognition network module, so
passing "ctc.qat_0711.no_outq.memristor_v15" resolves to this file, while the conversion and
prior jobs keep the plain v15 module and stay cached. The unmodified converted_model.pt is loaded
as-is; the out-quant observer keys become unexpected keys, which MiniReturnn tolerates
(load_state_dict with strict=False).
"""

from ..memristor_v15_mem_inited import *  # noqa: F401,F403
from ..memristor_v15_mem_inited import Model as _BaseModel
from ._strip import strip_output_quantizers


class Model(_BaseModel):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__(model_config_dict, **kwargs)
        strip_output_quantizers(self)
