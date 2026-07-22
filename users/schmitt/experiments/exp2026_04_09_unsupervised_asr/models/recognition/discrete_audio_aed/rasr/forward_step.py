from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Iterator, List, Literal, Optional, Protocol, Tuple, Callable

import torch
import returnn.frontend as rf
from returnn.tensor.tensor_dict import Tensor, TensorDict
from returnn.tensor import Dim, batch_dim
from sisyphus import Job, Task, tk


@dataclass
class DecoderConfigV1:
    search_function: Callable


class SearchFunction(Protocol):
    def __call__(self, features: torch.Tensor) -> Tuple[str, float]: ...


class AlignFunction(Protocol):
    def __call__(self, features: torch.Tensor, orth: str) -> Tuple[str, float]: ...


class EncoderModel(Protocol):
    def forward(self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor) -> torch.Tensor: ...


@lru_cache(maxsize=1)
def _get_rasr_search_function(config_file: tk.Path) -> SearchFunction:
    from librasr import Configuration, SearchAlgorithm

    config = Configuration()
    config.set_from_file(config_file)

    search_algorithm = SearchAlgorithm(config=config)

    def wrapper(features: torch.Tensor) -> Tuple[str, float]:
        nonlocal search_algorithm
        traceback = search_algorithm.recognize_segment(features)  # , seqTag="") # TODO
        recog_str = " ".join([traceback_item.lemma for traceback_item in traceback])
        if traceback != []:
            recog_score = traceback[-1].am_score + traceback[-1].lm_score
        else:
            recog_score = 0.0
        return recog_str, recog_score

    return wrapper


def forward_step_v1(
    *,
    model: EncoderModel,
    extern_data: TensorDict,
    search_function: SearchFunction,
    **unused_kwargs,
):
    data = extern_data["data"].raw_tensor
    seq_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    encoder_states, aux_logits, encoder_lens, _ = model.forward_audio(data, seq_len)

    batch_bytes = []
    batch_scores = []
    for b in range(data.size(0)):
        encoder_states_b = encoder_states[b].unsqueeze(0).to(device="cpu")  # [B, T, F]
        recog_str, recog_score = search_function(features=encoder_states_b)
        recog_str = recog_str.split()
        assert recog_str[-1] == "<EOS>"
        recog_str = " ".join(recog_str[:-1])
        batch_bytes.append(list(recog_str.encode("utf-8")))
        batch_scores.append(recog_score)

    byte_lens = list(map(len, batch_bytes))
    max_byte_len = max(byte_lens)
    seq_targets = torch.tensor(
        [bytes + [0] * (max_byte_len - len(bytes)) for bytes in batch_bytes],
        dtype=torch.uint8,
    )
    out_seq_len = torch.tensor(byte_lens, dtype=torch.int32)
    seq_log_prob = torch.tensor([score for score in batch_scores], dtype=torch.float32)

    bytes_dim = Dim(256, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, lens_dim], sparse_dim=bytes_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim])
