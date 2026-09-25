"""New in the port (replaces i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/eval_per.py
``_load_model`` / ``_decode_utt``: the fairseq forward of the §1c wav2vec-U 2.0 generator and its greedy decode).

RETURNN-side code of the GAN generator forward (the sisyphus side is :mod:`..analysis.w2vu2_gan_eval`):

* the model is the port's :class:`.recognizer.ConvRecognizer` at the §1c generator row
  (:data:`W2VU2_NET_ARGS`), built by :func:`.recognizer_only.get_model`.  ``ConvRecognizer`` already
  mirrors fairseq's ``Generator`` (``fairseq/examples/wav2vec/unsupervised/models/wav2vec_u.py``
  :288-367) layer for layer: BatchNorm1d over the non-padded frames only, residual
  ``x + in_proj(dropout(x))``, dropout, ``Conv1d(padding=kernel // 2, bias=False)``.  In eval mode
  the BN uses its running statistics and both dropouts are identities;
* :func:`generator_logits` -- the RAW generator logits ``[B, ceil(T / 3), 44]``.  The source's eval
  loaded the checkpoint with ``w2vu_generate.py``'s overrides ``segmentation.type = NONE`` (no
  segmenter pooling or join; the run TRAINED with ``JOIN``) and ``no_softmax = True``, so its logits
  are the generator conv output itself.  ``ConvRecognizer.forward`` returns a log-softmax, so the
  conv output is read with a forward hook on ``model.conv`` (the output layer at ``n_layers = 1``);
  no layer of the forward is re-implemented here;
* :func:`greedy_decode` -- the source's decode (fairseq's VITERBI decoder with its all-zero
  transition matrix): per-frame argmax, collapse consecutive-equal ids (``itertools.groupby``),
  drop the ``<SIL>`` index, map every other index to its dictionary symbol.  The four fairseq
  specials (``<s>``, ``<pad>``, ``</s>``, ``<unk>``) are kept if they win the argmax, as
  ``dictionary[i]`` kept them;
* :func:`w2vu2_forward_step` -- RETURNN ``forward_step``: marks the raw logits as ``logits``;
* :func:`W2vu2GreedyDecodeCallback` -- RETURNN ``forward_callback``: decodes each utterance with
  :func:`greedy_decode` and writes ``hyps.json`` (``{utterance id: [symbol, ...]}``, the collapsed,
  SIL-free strings the source scored and used as pseudo-labels) and ``hyps.stats.txt``.

Differences from the source that do not change a decode: the source ran one utterance per forward;
here utterances are batched, and ``ConvRecognizer`` zeroes the padded frames before the conv, so each
utterance's logits are its single-utterance logits up to float32 GEMM rounding.

Nothing here imports ``torch`` or ``returnn`` at module import time, so the sisyphus side can read
:data:`W2VU2_NET_ARGS` at graph time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

__all__ = [
    "W2VU2_NET_ARGS",
    "SIL_SYMBOL",
    "FAIRSEQ_SPECIAL_SYMBOLS",
    "LOGITS_KEY",
    "generator_logits",
    "greedy_decode",
    "w2vu2_forward_step",
    "W2vu2GreedyDecodeCallback",
]

#: The §1c generator row, read off the s0 GAN's own resolved config (``cfg["model"]`` of
#: ``FairseqW2vu2TrainJob.HOb2GgtYT7Bc/output/train/checkpoint_best.pt``): ``input_dim 1024,
#: generator_kernel 9, generator_stride 3, generator_dilation 1, generator_pad -1, generator_bias
#: False, generator_dropout 0.1, generator_batch_norm 30, generator_residual True``; 44 outputs =
#: its ``generator.proj.1.weight`` rows = the 4 fairseq specials + the 40 symbols of the text
#: side's ``dict.txt`` (39 ARPAbet + ``<SIL>``).  ``batch_norm = 30`` is only the BN weight's
#: initial value; the checkpoint overwrites it.
#: ``analysis.w2vu2_gan_eval.W2vu2GeneratorCheckpointJob`` checks every entry against the
#: checkpoint it converts.
W2VU2_NET_ARGS: Dict[str, Any] = {
    "in_dim": 1024,
    "n_out": 44,
    "kernel": 9,
    "stride": 3,
    "n_layers": 1,
    "dropout": 0.1,
    "batch_norm": 30.0,
    "residual": True,
    "bias": False,
}

#: the silence symbol of the GAN's text side (``dictionary.index("<SIL>")`` in the source decode)
SIL_SYMBOL = "<SIL>"
#: fairseq ``Dictionary()``'s specials, indices 0..3, ahead of the ``dict.txt`` symbols
FAIRSEQ_SPECIAL_SYMBOLS = ("<s>", "<pad>", "</s>", "<unk>")
#: the ``forward_step`` output key
LOGITS_KEY = "logits"


# -------------------------------------------------------------------------------------------------
# model forward
# -------------------------------------------------------------------------------------------------
def generator_logits(model, feats, lens):
    """``[B, T, 1024]`` features (fp16 or fp32) + ``[B]`` lengths -> ``[B, ceil(T / stride), n_out]``
    raw generator logits (fairseq's eval output with ``no_softmax``), float32.

    ``model`` is a ``ConvRecognizer`` with ``n_layers = 1``; its output conv's output is captured by
    a forward hook while ``ConvRecognizer.forward`` runs unchanged.
    """
    assert model.mid is None, "one conv layer expected: the hooked conv must be the output layer"
    captured = []
    handle = model.conv.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        model(feats, lens)
    finally:
        handle.remove()
    assert len(captured) == 1, len(captured)
    return captured[0].transpose(1, 2).float()  # [B, n_out, T_out] -> [B, T_out, n_out]


# -------------------------------------------------------------------------------------------------
# greedy decode (eval_per._decode_utt)
# -------------------------------------------------------------------------------------------------
def greedy_decode(logits, vocab: Sequence[str], sil_symbol: str = SIL_SYMBOL) -> List[str]:
    """``[T, V]`` logits -> the collapsed, SIL-free symbol list.

    argmax per frame (first index on ties, as ``torch.argmax``), collapse consecutive-equal ids,
    drop the ``sil_symbol`` index, map the rest through ``vocab`` (the fairseq dictionary order).
    """
    import itertools

    import numpy as np

    x = np.asarray(logits)
    assert x.ndim == 2 and x.shape[1] == len(vocab), (x.shape, len(vocab))
    sil = list(vocab).index(sil_symbol)
    ids = np.argmax(x, axis=1).tolist()
    collapsed = [k for k, _ in itertools.groupby(ids)]
    return [vocab[k] for k in collapsed if k != sil]


# -------------------------------------------------------------------------------------------------
# RETURNN forward step
# -------------------------------------------------------------------------------------------------
def w2vu2_forward_step(*, model, extern_data, features_key: str = "data", **_kwargs):
    """RETURNN ``forward_step``: mark the raw generator logits ``[B, T_out, n_out]`` as ``logits``.

    ``features_key`` is what a plain ``HDFDataset`` serves (``analysis.posterior_steps.FORWARD_DATA_KEY``).
    """
    import torch

    import returnn.frontend as rf
    from returnn.tensor import Dim, Tensor as ReturnnTensor

    assert not model.training, "the generator decode is an eval-mode forward (BN running stats, no dropout)"
    feats_ = extern_data[features_key]
    feats = feats_.raw_tensor
    b_dim = feats_.dims[0]
    feat_lens = feats_.dims[1].dyn_size_ext.raw_tensor.to(feats.device)

    with torch.no_grad():
        logits = generator_logits(model, feats, feat_lens)
    out_lens = model.output_lengths(feat_lens)
    assert int(logits.shape[1]) == int(out_lens.max()), (tuple(logits.shape), out_lens)

    t_dim = Dim(
        ReturnnTensor("out_lens", dims=[b_dim], dtype="int32", raw_tensor=out_lens.to(device="cpu", dtype=torch.int32)),
        name="gen_time",
    )
    v_dim = Dim(int(logits.shape[-1]), name="gen_vocab")
    rf.get_run_ctx().mark_as_output(
        rf.convert_to_tensor(logits.detach(), dims=[b_dim, t_dim, v_dim]), LOGITS_KEY, dims=[b_dim, t_dim, v_dim]
    )


# -------------------------------------------------------------------------------------------------
# RETURNN forward callback
# -------------------------------------------------------------------------------------------------
def W2vu2GreedyDecodeCallback(
    *,
    vocab_file: str,
    sil_symbol: str = SIL_SYMBOL,
    expected_num_seqs: int = 0,
    out_hyps_file: str = "hyps.json",
    out_stats_file: str = "hyps.stats.txt",
    **_kwargs,
):
    """Build the forward callback writing ``hyps.json`` ``{utterance id: [symbol, ...]}``.

    A factory, not a class (as ``analysis.posterior_steps.PosteriorHdfCallback``): RETURNN asserts
    the callback is a ``ForwardCallbackIface`` instance and calls ``forward_callback()`` when it is
    callable, so ``import returnn`` stays out of module import time.

    :param vocab_file: json list of the generator's output symbols in index order
        (``W2vu2GeneratorCheckpointJob.out_vocab``).
    :param expected_num_seqs: if > 0, ``finish`` asserts exactly this many utterances were decoded.
    """
    import json
    from collections import Counter

    from returnn.forward_iface import ForwardCallbackIface

    with open(str(vocab_file)) as fh:
        vocab = json.load(fh)
    assert sil_symbol in vocab, f"{sil_symbol!r} not in the generator vocabulary {vocab}"

    class _W2vu2GreedyDecodeCallback(ForwardCallbackIface):
        def __init__(self):
            self._hyps = {}
            self._frames = 0
            self._census = Counter()

        def init(self, *args, **kwargs):
            self._hyps = {}
            self._frames = 0
            self._census = Counter()

        def process_seq(self, *, seq_tag, outputs, **kwargs):
            import numpy as np

            x = np.asarray(outputs[LOGITS_KEY].raw_tensor, dtype="float32")
            assert x.ndim == 2 and x.shape[1] == len(vocab) and x.shape[0] > 0, (seq_tag, x.shape)
            assert seq_tag not in self._hyps, f"duplicate seq tag in forward: {seq_tag}"
            hyp = greedy_decode(x, vocab, sil_symbol)
            self._hyps[seq_tag] = hyp
            self._frames += int(x.shape[0])
            self._census.update(hyp)

        def finish(self, **kwargs):
            if expected_num_seqs > 0:
                assert len(self._hyps) == expected_num_seqs, (len(self._hyps), expected_num_seqs)
            with open(out_hyps_file, "w") as fh:
                json.dump({t: self._hyps[t] for t in sorted(self._hyps)}, fh)
            n = len(self._hyps)
            emitted = sum(len(h) for h in self._hyps.values())
            lines = [
                "wav2vec-U 2.0 generator greedy decode: argmax, collapse repeats, drop "
                f"{sil_symbol} (eval mode, segmentation NONE, no softmax)",
                f"utts = {n}   output frames = {self._frames}   phones emitted = {emitted}   "
                f"empty = {sum(1 for h in self._hyps.values() if not h)}",
                f"specials emitted = { {s: self._census[s] for s in FAIRSEQ_SPECIAL_SYMBOLS if self._census[s]} }",
                f"symbol types emitted = {len(self._census)} of {len(vocab) - 1} non-SIL",
            ]
            with open(out_stats_file, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            print("\n".join(lines), flush=True)

    return _W2vu2GreedyDecodeCallback()
