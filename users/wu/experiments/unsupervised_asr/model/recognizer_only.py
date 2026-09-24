"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/init_jobs.py (``RECOGNIZER_NET_ARGS``,
lines 93-104, and the RETURNN ``get_model``, lines 458-477, verbatim but for the relative import).

The recognizer-only model: a bare ``ConvRecognizer`` as RETURNN's ``get_model``, used by the
flat recognizer init and by the posterior dump.  The init job itself and its CTC ``train_step``
are not part of this module.
"""

from typing import Any, Dict

__all__ = ["RECOGNIZER_NET_ARGS", "get_model"]

# Model shape: the §1c generator row, read off FairseqW2vu2TrainJob.HOb2GgtYT7Bc's resolved config,
# with the two amendments of SAE_4A.md:66-81 (stride 1, 41 outputs).  See recognizer.py.
RECOGNIZER_NET_ARGS: Dict[str, Any] = {
    "in_dim": 1024,      # setup report §2: w2v2-Large-lv60 layer 15, 1024-d @ 50 Hz
    "n_out": 41,         # SAE_4A.md:66-67: blank + 39 ARPAbet + SIL
    "kernel": 9,         # §1c generator_kernel
    "stride": 1,         # SAE_4A.md:74-79 amendment (§1c strides 3; 2 is the registered ablation)
    "hidden": 512,
    "n_layers": 1,       # §1c generator depth
    "dropout": 0.1,      # §1c generator_dropout
    "batch_norm": 30.0,  # §1c generator_batch_norm
    "residual": True,    # §1c generator_residual
    "bias": False,       # §1c generator_bias
}


def get_model(*, epoch: int, step: int, **kwargs):
    """RETURNN ``get_model``: the bare ConvRecognizer (``net_args`` are bound by PartialImport).

    RETURNN calls this as ``get_model(epoch=..., step=..., device=..., **sentinel_kw)``
    (``returnn/torch/engine.py:1242-1245``), where ``sentinel_kw`` is a randomly named
    forward-compatibility kwarg (``returnn/util/basic.py:4552-4558``).  None of those are model
    hyper-parameters, and ``ConvRecognizer.__init__`` rejects them.  The shim -- not the module --
    absorbs them: anything the recognizer's signature does not declare is dropped and PRINTED, so
    a silently ignored argument is still visible in the training log.
    """
    import inspect

    from .recognizer import ConvRecognizer

    declared = set(inspect.signature(ConvRecognizer.__init__).parameters) - {"self"}
    net_args = {k: v for k, v in kwargs.items() if k in declared}
    dropped = sorted(set(kwargs) - set(net_args))
    if dropped:
        print(f"get_model: ignoring kwargs not declared by ConvRecognizer: {dropped}", flush=True)
    return ConvRecognizer(**net_args)
