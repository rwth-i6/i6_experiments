"""
RETURNN Forward / Beam Search Decoding Step for UNdreaMT (src -> trg and trg -> src).
Produces search_out.py.gz compatible with SearchTakeBestJob and ComputeBleuJob.
"""

from typing import Dict, Optional
import os
import torch

from returnn.tensor import Tensor, Dim, TensorDict
import returnn.frontend as rf

from ...definitions.undreamt_rnn_aed_v1 import UNdreaMTModel
from ...train_steps.undreamt_backtranslation_train_step import prepare_undreamt_tensors


_beam_size_param: int = 12
_lp_alpha_param: float = 0.6
_max_ratio_param: float = 2.0
_src_lang_param: str = "src"
_trg_lang_param: str = "trg"
_input_data_key_param: Optional[str] = None


def forward_init_hook(
    run_ctx,
    *,
    beam_size: int = 12,
    lp_alpha: float = 0.6,
    max_ratio: float = 2.0,
    src_lang: str = "src",
    trg_lang: str = "trg",
    input_modality: Optional[str] = None,
    output_modality: Optional[str] = None,
    input_data_key: Optional[str] = None,
    **kwargs,
):
    global _beam_size_param, _lp_alpha_param, _max_ratio_param
    global _src_lang_param, _trg_lang_param, _input_data_key_param

    _beam_size_param = beam_size
    _lp_alpha_param = lp_alpha
    _max_ratio_param = max_ratio
    _src_lang_param = input_modality if input_modality is not None else src_lang
    _trg_lang_param = output_modality if output_modality is not None else trg_lang
    _input_data_key_param = input_data_key


def forward_finish_hook(run_ctx, **kwargs):
    pass


def forward_step(
    *,
    model: UNdreaMTModel,
    extern_data: TensorDict,
    beam_size: Optional[int] = None,
    lp_alpha: Optional[float] = None,
    max_ratio: Optional[float] = None,
    src_lang: Optional[str] = None,
    trg_lang: Optional[str] = None,
    input_modality: Optional[str] = None,
    output_modality: Optional[str] = None,
    input_data_key: Optional[str] = None,
    **kwargs,
):
    rf_ctx = rf.get_run_ctx()

    eff_beam_size = beam_size if beam_size is not None else _beam_size_param
    eff_lp_alpha = lp_alpha if lp_alpha is not None else _lp_alpha_param
    eff_max_ratio = max_ratio if max_ratio is not None else _max_ratio_param
    eff_src_lang = input_modality or src_lang or _src_lang_param
    eff_trg_lang = output_modality or trg_lang or _trg_lang_param
    data_key = input_data_key if input_data_key is not None else _input_data_key_param

    if data_key is None or data_key not in extern_data.data:
        data_key = "data" if "data" in extern_data.data else list(extern_data.data.keys())[0]

    input_tensor = extern_data[data_key]
    batch_dims = input_tensor.remaining_dims(remove=(input_tensor.dims[1],))

    raw_seqs = input_tensor.raw_tensor.long()
    raw_lens = input_tensor.dims[1].dyn_size_ext.raw_tensor.long().to(raw_seqs.device)

    enc_ids, enc_lens, _, _ = prepare_undreamt_tensors(
        raw_seqs, raw_lens, apply_swap_noise=False
    )

    hyps = model.beam_search_translate(
        src_ids=enc_ids,
        src_lengths=enc_lens,
        src_lang=eff_src_lang,
        trg_lang=eff_trg_lang,
        beam_size=eff_beam_size,
        max_ratio=eff_max_ratio,
        lp_alpha=eff_lp_alpha,
    )

    _, _, _, _, vocab_total = model.get_lang_modules(eff_trg_lang)
    vocab_dim = Dim(vocab_total, name="vocab")

    batch_size = len(hyps)
    beam_dim = Dim(1, name="beam")
    lengths = [len(h[0]) for h in hyps]
    max_length = max(lengths) if lengths else 0

    hyp_array = torch.zeros((batch_size, 1, max_length), dtype=torch.int32)
    scores_array = torch.zeros((batch_size, 1), dtype=torch.float32)

    for b_idx, (tokens, score) in enumerate(hyps):
        if len(tokens) > 0:
            hyp_array[b_idx, 0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        scores_array[b_idx, 0] = float(score)

    lengths_tensor = rf.convert_to_tensor(torch.tensor(lengths, dtype=torch.int32).unsqueeze(1), dims=batch_dims + [beam_dim])
    out_spatial_dim = Dim(lengths_tensor, name="out_spatial")

    rf_tokens = rf.convert_to_tensor(hyp_array, dims=batch_dims + [beam_dim, out_spatial_dim], sparse_dim=vocab_dim)
    rf_scores = rf.convert_to_tensor(scores_array, dims=batch_dims + [beam_dim])

    rf_ctx.mark_as_output(rf_tokens, "tokens", dims=batch_dims + [beam_dim, out_spatial_dim])
    rf_ctx.mark_as_output(rf_scores, "scores", dims=batch_dims + [beam_dim])

