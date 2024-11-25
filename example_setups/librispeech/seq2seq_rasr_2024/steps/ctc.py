from time import perf_counter
from typing import Protocol

import numpy as np
import torch
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import Tensor, TensorDict

from ..pytorch_models.ctc import ConformerCTCModel, ConformerCTCRecogModel


def train_step(*, model: ConformerCTCModel, extern_data: TensorDict, **_):
    raw_data = extern_data.as_raw_tensor_dict()
    audio_samples = raw_data["data"]  # [B, T, 1]
    audio_samples_size = raw_data["data:size1"]  # [B]

    targets = raw_data["classes"].long()  # [B, S]

    targets_size_rf = extern_data["classes"].dims[1].dyn_size_ext  # [B]
    assert targets_size_rf is not None
    targets_size = targets_size_rf.raw_tensor  # [B]
    assert targets_size is not None

    log_probs, log_probs_size = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size.to("cuda"),
    )  # [B, T, V], [B]

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, V]

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=log_probs_size,
        target_lengths=targets_size,
        blank=model.target_size - 1,
        reduction="sum",
        zero_infinity=True,
    )

    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    rf.get_run_ctx().mark_as_loss(
        name="CTC", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_size_rf, axis=batch_dim)
    )


class ComputePriorCallback(ForwardCallbackIface):
    def init(self, *, model: ConformerCTCModel):
        self.n = 1
        self.avg_probs = None

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        log_prob_tensor = outputs["log_probs"].raw_tensor
        assert log_prob_tensor is not None
        prob_tensor_iter = iter(np.exp(log_prob_tensor))

        if self.avg_probs is None:
            self.avg_probs = next(prob_tensor_iter)
            print("Create probs collection tensor of shape", self.avg_probs.shape)

        for prob_tensor in prob_tensor_iter:
            self.n += 1
            self.avg_probs += (prob_tensor - self.avg_probs) / self.n

    def finish(self):
        prob_array = self.avg_probs
        log_prob_array = np.log(prob_array)  # type: ignore
        log_prob_strings = ["%.20e" % s for s in log_prob_array]

        # Write txt file
        with open("prior.txt", "wt") as f:
            f.write(" ".join(log_prob_strings))

        # Write xml file
        with open("prior.xml", "wt") as f:
            f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="{len(log_prob_array)}">\n')
            f.write(" ".join(log_prob_strings))
            f.write("\n</vector-f32>")

        # Plot png file
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xdata = range(len(prob_array))  # type: ignore
        plt.semilogy(xdata, prob_array)
        plt.xlabel("emission idx")
        plt.ylabel("prior")
        plt.grid(True)
        plt.savefig("prior.png")


def prior_step(*, model: ConformerCTCModel, extern_data: TensorDict, **_):
    raw_data = extern_data.as_raw_tensor_dict()
    audio_samples = raw_data["data"]  # [B, T, 1]
    audio_samples_size = raw_data["data:size1"]  # [B]

    log_probs, sequence_lengths = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size.to("cuda"),
    )  # [B, T, V], [B]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        assert run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext is not None
        run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext.raw_tensor = sequence_lengths
    run_ctx.mark_as_output(log_probs, name="log_probs")


class SearchCallback(ForwardCallbackIface):
    def init(self, *, model: ConformerCTCRecogModel):
        self.total_audio_samples = 0
        self.total_am_time = 0
        self.total_search_time = 0

        self.recognition_file = open("search_out.py", "w")
        self.recognition_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        raw_outputs = outputs.as_raw_tensor_dict()
        token_seq = raw_outputs["tokens"]
        token_str = " ".join(token_seq)
        self.recognition_file.write(f"{repr(seq_tag)}: {repr(token_str)},\n")

        self.total_audio_samples += raw_outputs["audio_samples_size"]
        self.total_am_time += raw_outputs["am_time"]
        self.total_search_time += raw_outputs["search_time"]

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()

        total_audio_seconds = self.total_audio_samples / 16000

        print("Total AM time: %.2fs, AM-RTF: %.3f" % (self.total_am_time, self.total_am_time / total_audio_seconds))
        print(
            "Total search time: %.2fs, search-RTF: %.3f"
            % (self.total_search_time, self.total_search_time / total_audio_seconds)
        )

        total_time = self.total_am_time + self.total_search_time
        print("Total time: %.2f, RTF: %.3f" % (total_time, total_time / total_audio_seconds))


class SearchFunction(Protocol):
    def __call__(self, features: torch.Tensor) -> str: ...


def rasr_recog_step(
    *,
    model: ConformerCTCRecogModel,
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
    audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
    seq_tags = raw_data["seq_tag"]

    tokens_arrays = []
    token_lengths = []

    am_times = []
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

        am_start = perf_counter()
        model_outputs = model.forward(audio_samples=seq_samples, audio_samples_size=seq_samples_size)
        am_time = perf_counter() - am_start
        am_times.append(am_time)

        model_outputs = model_outputs.to(device="cpu")

        search_start = perf_counter()
        result: str = search_function(features=model_outputs)
        search_time = perf_counter() - search_start
        search_times.append(search_time)

        result = result.replace("<blank>", "")

        tokens_array = np.array(result.split(), dtype="U")
        tokens_arrays.append(tokens_array)
        token_lengths.append(len(tokens_array))

        seq_time = seq_samples_size[0] / 16000

        print(f"Recognized sequence {repr(seq_tags[b])}")
        print(f"    AM time: {am_time:.3f} seconds, RTF {am_time / seq_time:.3f}")
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

    am_time_tensor = Tensor(
        name="am_time",
        dtype="float32",
        raw_tensor=np.array(am_times, dtype=np.float32),
        feature_dim_axis=None,
        time_dim_axis=None,
    )
    run_ctx.mark_as_output(am_time_tensor, name="am_time")

    search_time_tensor = Tensor(
        name="search_time",
        dtype="float32",
        raw_tensor=np.array(search_times, dtype=np.float32),
        feature_dim_axis=None,
        time_dim_axis=None,
    )
    run_ctx.mark_as_output(search_time_tensor, name="search_time")
