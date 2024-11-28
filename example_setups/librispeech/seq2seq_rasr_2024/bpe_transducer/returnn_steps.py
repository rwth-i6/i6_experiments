from functools import lru_cache
from time import perf_counter
from typing import Dict, Iterator, Optional, Protocol, Union

import numpy as np
import torch
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import Tensor, TensorDict
from sisyphus import Job, Task, tk

from .pytorch_modules import (
    FFNNTransducerCTCModel,
    FFNNTransducerCTCRecogModel,
    FFNNTransducerEncoder,
    FFNNTransducerModel,
    FFNNTransducerScorer,
)


def train_step(*, model: FFNNTransducerModel, extern_data: TensorDict, enc_loss_scales: Dict[int, float] = {}, **_):
    import returnn.frontend as rf
    from i6_native_ops.monotonic_rnnt import monotonic_rnnt_loss
    from returnn.tensor import batch_dim

    run_ctx = rf.get_run_ctx()

    raw_data = extern_data.as_raw_tensor_dict()

    audio_samples = raw_data["data"]  # [B, T, 1]
    audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)  # [B]

    targets = raw_data["classes"]  # [B, S]

    targets_size_rf = extern_data["classes"].dims[1].dyn_size_ext  # [B]
    assert targets_size_rf is not None
    targets_size = targets_size_rf.raw_tensor  # [B]
    assert targets_size is not None
    targets_size = targets_size.to(device=audio_samples.device)

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


class ComputeCTCPriorCallback(ForwardCallbackIface):
    def init(self, *, model: FFNNTransducerCTCModel):
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


def ctc_prior_step(*, model: FFNNTransducerCTCModel, extern_data: TensorDict, **_):
    raw_data = extern_data.as_raw_tensor_dict()
    audio_samples = raw_data["data"]  # [B, T, 1]
    audio_samples_size = raw_data["data:size1"]  # [B]

    log_probs, sequence_lengths = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size.to(device=audio_samples.device),
    )  # [B, T, V], [B]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        assert run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext is not None
        run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext.raw_tensor = sequence_lengths
    run_ctx.mark_as_output(log_probs, name="log_probs")


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
    def init(self, *, model: Union[FFNNTransducerScorer, FFNNTransducerCTCRecogModel]):
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

        with open("rtf.py", "w") as rtf_file:
            rtf_file.write("{\n")
            total_audio_seconds = self.total_audio_samples / 16000
            rtf_file.write(f'    "audio_seconds": {total_audio_seconds},\n')

            enc_rtf = self.total_enc_time / total_audio_seconds
            enc_rtfx = total_audio_seconds / self.total_enc_time

            print(f"Total encoder time: {self.total_enc_time:.2f} seconds, AM-RTF: {enc_rtf}, XRTF: {enc_rtfx}")

            rtf_file.write(f'    "enc_seconds": {self.total_enc_time},\n')
            rtf_file.write(f'    "enc_rtf": {enc_rtf},\n')
            rtf_file.write(f'    "enc_rtfx": {enc_rtfx},\n')

            search_rtf = self.total_search_time / total_audio_seconds
            search_rtfx = total_audio_seconds / self.total_search_time

            print(
                f"Total search time: {self.total_search_time:.2f} seconds, search-RTF: {search_rtf}, RTFX: {search_rtfx}"
            )
            rtf_file.write(f'    "search_seconds": {self.total_search_time},\n')
            rtf_file.write(f'    "search_rtf": {search_rtf},\n')
            rtf_file.write(f'    "search_rtfx": {search_rtfx},\n')

            total_time = self.total_enc_time + self.total_search_time
            total_rtf = total_time / total_audio_seconds
            total_rtfx = total_audio_seconds / total_time

            print(f"Total time: {total_time:.2f} seconds, RTF: {total_rtf}, RTFX: {total_rtfx}")
            rtf_file.write(f'    "total_seconds": {total_time},\n')
            rtf_file.write(f'    "total_rtf": {total_rtf},\n')
            rtf_file.write(f'    "total_rtfx": {total_rtfx},\n')

            rtf_file.write("}\n")


class ExtractSearchRTFJob(Job):
    def __init__(self, rtf_file: tk.Path) -> None:
        self.rtf_file = rtf_file

        self.out_audio_seconds = self.output_var("audio_seconds")
        self.out_enc_seconds = self.output_var("enc_seconds")
        self.out_enc_rtf = self.output_var("enc_rtf")
        self.out_enc_rtfx = self.output_var("enc_rtfx")
        self.out_search_seconds = self.output_var("search_seconds")
        self.out_search_rtf = self.output_var("search_rtf")
        self.out_search_rtfx = self.output_var("search_rtfx")
        self.out_total_seconds = self.output_var("total_seconds")
        self.out_total_rtf = self.output_var("total_rtf")
        self.out_total_rtfx = self.output_var("total_rtfx")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.rtf_file.get(), "r") as f:
            result_dict = eval(f.read())
            self.out_audio_seconds.set(result_dict["audio_seconds"])
            self.out_enc_seconds.set(result_dict["enc_seconds"])
            self.out_enc_rtf.set(result_dict["enc_rtf"])
            self.out_enc_rtfx.set(result_dict["enc_rtfx"])
            self.out_search_seconds.set(result_dict["search_seconds"])
            self.out_search_rtf.set(result_dict["search_rtf"])
            self.out_search_rtfx.set(result_dict["search_rtfx"])
            self.out_total_seconds.set(result_dict["total_seconds"])
            self.out_total_rtf.set(result_dict["total_rtf"])
            self.out_total_rtfx.set(result_dict["total_rtfx"])


class SearchFunction(Protocol):
    def __call__(self, features: torch.Tensor) -> str: ...


@lru_cache(maxsize=1)
def get_rasr_search_function(config_file: tk.Path) -> SearchFunction:
    from librasr import Configuration, SearchAlgorithm

    config = Configuration()
    config.set_from_file(config_file)

    search_algorithm = SearchAlgorithm(config=config)
    return search_algorithm.recognize_segment


@lru_cache(maxsize=1)
def get_ctc_flashlight_search_function(
    vocab_file: tk.Path,
    lexicon_file: Optional[str],
    lm_file: Optional[str],
    beam_size: int,
    beam_size_token: Optional[int],
    beam_threshold: float,
    lm_scale: float,
) -> SearchFunction:
    from torchaudio.models.decoder import ctc_decoder

    vocab = Vocabulary.create_vocab(vocab_file=vocab_file, unknown_label=None)
    assert vocab._vocab is not None
    labels = list({value: key for key, value in vocab._vocab.items()}.values())

    if "" not in labels:
        labels.append("")

    print(f"labels: {labels}")

    decoder = ctc_decoder(
        lexicon=lexicon_file,
        tokens=labels,
        lm=lm_file,
        nbest=1,
        beam_size=beam_size,
        beam_size_token=beam_size_token,
        beam_threshold=beam_threshold,
        lm_weight=lm_scale,
        sil_score=float("inf"),
        blank_token="<blank>",
        sil_token="",
        unk_word="<unk>",
    )

    def wrapper(features: torch.Tensor) -> str:
        nonlocal labels
        nonlocal decoder

        hyps = decoder(-features)
        str_result = " ".join([labels[token] for token in hyps[0][0].tokens])
        return str_result

    return wrapper


def recog_step(
    *,
    model: Union[FFNNTransducerEncoder, FFNNTransducerCTCRecogModel],
    extern_data: TensorDict,
    search_function: SearchFunction,
    **_,
):
    raw_data = extern_data.as_raw_tensor_dict()
    audio_samples = raw_data["data"]
    audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
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
        print(
            f"    Encoder time: {encoder_time:.3f} seconds, RTF {encoder_time / seq_time:.3f}, XRTF {seq_time / encoder_time:.3f}"
        )
        print(
            f"    Search time: {search_time:.3f} seconds, RTF {search_time / seq_time:.3f}, XRTF {seq_time / search_time:.3f}"
        )
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


def rasr_recog_step(
    *,
    model: Union[FFNNTransducerEncoder, FFNNTransducerCTCRecogModel],
    extern_data: TensorDict,
    config_file: tk.Path,
    **kwargs,
):
    search_function = get_rasr_search_function(config_file=config_file)
    return recog_step(model=model, extern_data=extern_data, search_function=search_function, **kwargs)


def ctc_flashlight_recog_step(
    *,
    model: FFNNTransducerCTCRecogModel,
    extern_data: TensorDict,
    vocab_file: tk.Path,
    lexicon_file: Optional[str],
    lm_file: Optional[str],
    beam_size: int,
    beam_size_token: Optional[int],
    beam_threshold: float,
    lm_scale: float,
    **kwargs,
):
    search_function = get_ctc_flashlight_search_function(
        vocab_file=vocab_file,
        lexicon_file=lexicon_file,
        lm_file=lm_file,
        beam_size=beam_size,
        beam_size_token=beam_size_token,
        beam_threshold=beam_threshold,
        lm_scale=lm_scale,
    )
    return recog_step(model=model, extern_data=extern_data, search_function=search_function, **kwargs)
