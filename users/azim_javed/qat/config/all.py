from sisyphus import tk

from .digital.ffnn_transducer import all as digital_ffnn
from .digital.full_ctx_transducer import all as digital_full_ctx
from .digital.ctc import all as digital_ctc
from .memristor.ffnn_transducer import all as memristor_ffnn
from .memristor.full_ctx_transducer import all as memristor_full_ctx
from .memristor.ctc import all as memristor_ctc


async def py():
    digital_ffnn.py()
    digital_full_ctx.py()
    digital_ctc.py()
    await tk.async_gather(memristor_ffnn.py(), memristor_full_ctx.py(), memristor_ctc.py())
