from time import perf_counter
from typing import Dict, Protocol

import numpy as np
import torch
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import Tensor, TensorDict

from ..pytorch_models.transducer import FFNNTransducerEncoder, FFNNTransducerModel, FFNNTransducerScorer


def train_step(*, model: FFNNTransducerModel, extern_data: TensorDict, enc_loss_scales: Dict[int, float] = {}, **_):
    import returnn.frontend as rf
    from i6_native_ops.monotonic_rnnt import monotonic_rnnt_loss
    from returnn.tensor import batch_dim

    run_ctx = rf.get_run_ctx()

    raw_data = extern_data.as_raw_tensor_dict()

    audio_samples = raw_data["data"]  # [B, T, 1]
    audio_samples_size = raw_data["data:size1"].to(device="cuda")  # [B]

    targets = raw_data["classes"]  # [B, S]

    targets_size_rf = extern_data["classes"].dims[1].dyn_size_ext  # [B]
    assert targets_size_rf is not None
    targets_size = targets_size_rf.raw_tensor  # [B]
    assert targets_size is not None
    targets_size = targets_size.to(device="cuda")

    logits, intermediate_encoder_log_probs, encoder_states_size = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size,
        targets=targets,
        targets_size=targets_size,
    )

    rnnt_loss = monotonic_rnnt_loss(
        acts=logits,
        labels=targets,
        input_lengths=encoder_states_size,
        label_lengths=targets_size,
        blank_label=model.target_size - 1,
    ).sum()
    loss_norm_factor = rf.reduce_sum(targets_size_rf, axis=batch_dim)

    run_ctx.mark_as_loss(name="MonoRNNT", loss=rnnt_loss, custom_inv_norm_factor=loss_norm_factor)

    for layer_idx, scale in enc_loss_scales.items():
        log_probs = intermediate_encoder_log_probs[layer_idx]  # [B, T, V]

        log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, C]
        loss = torch.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=encoder_states_size,
            target_lengths=targets_size,
            blank=model.target_size - 1,
            reduction="sum",
            zero_infinity=True,
        )

        rf.get_run_ctx().mark_as_loss(
            name=f"CTC_enc-{layer_idx}",
            loss=loss,
            scale=scale,
            custom_inv_norm_factor=loss_norm_factor,
        )


def scorer_forward_step(*, model: FFNNTransducerScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor  # [B, F]
    assert encoder_state is not None
    history = extern_data["history"].raw_tensor  # [B, S]
    assert history is not None

    scores = model.forward(
        encoder_state=encoder_state,
        history=history,
    )

    run_ctx.mark_as_output(name="scores", tensor=scores)


class SearchCallback(ForwardCallbackIface):
    def init(self, *, model: FFNNTransducerEncoder):
        self.total_audio_samples = 0
        self.total_enc_time = 0
        self.total_search_time = 0

        self.recognition_file = open("search_out.py", "w")
        self.recognition_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        raw_outputs = outputs.as_raw_tensor_dict()
        token_seq = raw_outputs["tokens"]
        token_str = " ".join(token_seq)
        self.recognition_file.write(f"{repr(seq_tag)}: {repr(token_str)},\n")

        self.total_audio_samples += raw_outputs["audio_samples_size"]
        self.total_enc_time += raw_outputs["enc_time"]
        self.total_search_time += raw_outputs["search_time"]

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()

        total_audio_seconds = self.total_audio_samples / 16000

        print(
            "Total encoder time: %.2fs, encoder-RTF: %.3f"
            % (self.total_enc_time, self.total_enc_time / total_audio_seconds)
        )
        print(
            "Total search time: %.2fs, search-RTF: %.3f"
            % (self.total_search_time, self.total_search_time / total_audio_seconds)
        )

        total_time = self.total_enc_time + self.total_search_time
        print("Total time: %.2f, RTF: %.3f" % (total_time, total_time / total_audio_seconds))


class SearchFunction(Protocol):
    def __call__(self, features: torch.Tensor) -> str: ...


def rasr_recog_step(
    *,
    model: FFNNTransducerEncoder,
    extern_data: TensorDict,
    # search_function: SearchFunction,
    config_file: str,
    **_,
):
    from librasr import Configuration, SearchAlgorithm

    config = Configuration()
    config.set_from_file(config_file)
    search_algorithm = SearchAlgorithm(config=config)
    search_function = search_algorithm.recognize_segment

    raw_data = extern_data.as_raw_tensor_dict()
    audio_samples = raw_data["data"]
    audio_samples_size = raw_data["data:size1"].to(device="cuda")
    seq_tags = raw_data["seq_tag"]

    tokens_arrays = []
    token_lengths = []

    encoder_times = []
    search_times = []

    search_function(
        features=model.forward(
            torch.zeros(1, 1000, 1, device=audio_samples.device),
            audio_samples_size=torch.full([1], fill_value=1000, device=audio_samples.device),
        ).to(device="cpu")
    )

    for b in range(audio_samples.size(0)):
        seq_samples_size = audio_samples_size[b : b + 1]
        seq_samples = audio_samples[b : b + 1, : seq_samples_size[0]]  # [1, T, 1]

        encoder_start = perf_counter()
        encoder_states = model.forward(audio_samples=seq_samples, audio_samples_size=seq_samples_size)
        encoder_time = perf_counter() - encoder_start
        encoder_times.append(encoder_time)

        encoder_states = encoder_states.to(device="cpu")

        search_start = perf_counter()
        result: str = search_function(features=encoder_states)
        search_time = perf_counter() - search_start
        search_times.append(search_time)

        result = result.replace("<blank>", "")

        tokens_array = np.array(result.split(), dtype="U")
        tokens_arrays.append(tokens_array)
        token_lengths.append(len(tokens_array))

        seq_time = seq_samples_size[0] / 16000

        print(f"Recognized sequence {repr(seq_tags[b])}")
        print(f"    Encoder time: {encoder_time:.3f} seconds, RTF {encoder_time / seq_time:.3f}")
        print(f"    Search time: {search_time:.3f} seconds, RTF {search_time / seq_time:.3f}")
        print(f"    Tokens: {tokens_array}")
        print()

    max_len = np.max(token_lengths)
    tokens_arrays_padded = [
        np.pad(tokens_array, pad_width=(0, max_len - len(tokens_array))) for tokens_array in tokens_arrays
    ]

    tokens_tensor = Tensor(
        name="tokens", dtype="string", raw_tensor=np.stack(tokens_arrays_padded, axis=0), feature_dim_axis=None
    )
    tokens_len_array = np.array(token_lengths, dtype=np.int32)

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        assert run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext is not None
        run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext.raw_tensor = tokens_len_array
    run_ctx.mark_as_output(tokens_tensor, name="tokens")

    audio_samples_tensor = Tensor(
        name="audio_samples_size",
        dtype="int32",
        raw_tensor=audio_samples_size,
        feature_dim_axis=None,
        time_dim_axis=None,
    )
    run_ctx.mark_as_output(audio_samples_tensor, name="audio_samples_size")

    enc_time_tensor = Tensor(
        name="enc_time",
        dtype="float32",
        raw_tensor=np.array(encoder_times, dtype=np.float32),
        feature_dim_axis=None,
        time_dim_axis=None,
    )
    run_ctx.mark_as_output(enc_time_tensor, name="enc_time")

    search_time_tensor = Tensor(
        name="search_time",
        dtype="float32",
        raw_tensor=np.array(search_times, dtype=np.float32),
        feature_dim_axis=None,
        time_dim_axis=None,
    )
    run_ctx.mark_as_output(search_time_tensor, name="search_time")
