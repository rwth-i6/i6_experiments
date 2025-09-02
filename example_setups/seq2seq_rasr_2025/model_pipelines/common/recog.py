__all__ = [
    "RecogResult",
    "OfflineRecogResult",
    "OfflineRecogResultWithSearchErrors",
    "StreamingRecogResult",
    "recog_rasr_offline",
    "recog_rasr_offline_with_search_errors",
    "recog_rasr_streaming",
]

from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, List, Protocol

import numpy as np
import torch
from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchBPEtoWordsJob, SearchWordsToCTMJob
from i6_core.util import DelayedFormat
from i6_experiments.common.setups.serialization import Call, Collection, ExternalImport, Import
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import Tensor, TensorDict
from sisyphus import Job, Task, tk

from ...data.base import DataConfig
from ...tools import rasr_binary_path, returnn_python_exe, returnn_root
from .corpus import ScorableCorpus
from .serializers import recipe_imports

# ========================================
# ========== Result Dataclasses ==========
# ========================================


@dataclass
class RecogResult:
    descriptor: str
    corpus_name: str
    wer: tk.Variable
    deletion: tk.Variable
    insertion: tk.Variable
    substitution: tk.Variable


@dataclass
class OfflineRecogResult(RecogResult):
    enc_rtf: tk.Variable
    search_rtf: tk.Variable
    total_rtf: tk.Variable


@dataclass
class OfflineRecogResultWithSearchErrors(OfflineRecogResult):
    search_error_rate: tk.Variable
    model_error_rate: tk.Variable
    skipped_rate: tk.Variable
    correct_rate: tk.Variable


@dataclass
class StreamingRecogResult(RecogResult):
    unstable_latency_avg: tk.Variable
    unstable_latency_p50: tk.Variable
    unstable_latency_p90: tk.Variable
    unstable_latency_p99: tk.Variable
    unstable_latency_p100: tk.Variable
    stable_latency_avg: tk.Variable
    stable_latency_p50: tk.Variable
    stable_latency_p90: tk.Variable
    stable_latency_p99: tk.Variable
    stable_latency_p100: tk.Variable


# =======================================================
# ========== Callbacks for result file writing ==========
# =======================================================


class SearchCallback(ForwardCallbackIface):
    def init(self, *, model):
        self.recognition_file = open("search_out.py", "w")
        self.recognition_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        raw_outputs = outputs.as_raw_tensor_dict()
        token_seq = raw_outputs["tokens"]
        token_str = " ".join(token_seq)
        self.recognition_file.write(f"{repr(seq_tag)}: {repr(token_str)},\n")

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()


class OfflineSearchCallback(SearchCallback):
    def init(self, *, model):
        super().init(model=model)

        self.total_audio_samples = 0
        self.total_enc_time = 0
        self.total_search_time = 0
        self.total_seqs = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)

        self.total_seqs += 1
        raw_outputs = outputs.as_raw_tensor_dict()

        self.total_audio_samples += raw_outputs["audio_samples_size"]
        self.total_enc_time += raw_outputs["enc_time"]
        self.total_search_time += raw_outputs["search_time"]

    def finish(self):
        super().finish()

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


class OfflineSearchCallbackWithSearchErrors(OfflineSearchCallback):
    def init(self, *, model):
        super().init(model=model)
        self.total_skipped = 0
        self.total_correct = 0
        self.total_search_errors = 0
        self.total_model_errors = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)
        raw_outputs = outputs.as_raw_tensor_dict()

        self.total_skipped += raw_outputs["skipped"]
        self.total_correct += raw_outputs["correct"]
        self.total_search_errors += raw_outputs["search_errors"]
        self.total_model_errors += raw_outputs["model_errors"]

    def finish(self):
        super().finish()

        with open("search_errors.py", "w") as search_error_file:
            search_error_file.write("{\n")
            search_error_file.write(f'    "total_skipped": {self.total_skipped},\n')
            search_error_file.write(f'    "skipped_rate": {self.total_skipped / self.total_seqs},\n')
            search_error_file.write(f'    "total_correct": {self.total_correct},\n')
            search_error_file.write(f'    "correct_rate": {self.total_correct / self.total_seqs},\n')
            search_error_file.write(f'    "total_search_errors": {self.total_search_errors},\n')
            search_error_file.write(f'    "search_error_rate": {self.total_search_errors / self.total_seqs},\n')
            search_error_file.write(f'    "total_model_errors": {self.total_model_errors},\n')
            search_error_file.write(f'    "model_error_rate": {self.total_model_errors / self.total_seqs},\n')
            search_error_file.write("}\n")


class StreamingSearchCallback(SearchCallback):
    def init(self, *, model):
        super().init(model=model)

        self.unstable_latencies = []
        self.stable_latencies = []

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)

        raw_outputs = outputs.as_raw_tensor_dict()

        for latency in raw_outputs["unstable_latencies"]:
            self.unstable_latencies.append(latency)

        for latency in raw_outputs["stable_latencies"]:
            self.stable_latencies.append(latency)

    def finish(self):
        super().finish()

        import numpy as np

        with open("latencies.py", "w") as latency_file:
            latency_file.write("{\n")

            unstable_np_array = np.array(self.unstable_latencies)
            stable_np_array = np.array(self.stable_latencies)

            latency_file.write(f'    "unstable_latency_avg": {np.average(unstable_np_array)},\n')
            latency_file.write(f'    "unstable_latency_p50": {np.percentile(unstable_np_array, 50)},\n')
            latency_file.write(f'    "unstable_latency_p90": {np.percentile(unstable_np_array, 90)},\n')
            latency_file.write(f'    "unstable_latency_p99": {np.percentile(unstable_np_array, 99)},\n')
            latency_file.write(f'    "unstable_latency_p100": {np.percentile(unstable_np_array, 100)},\n')
            latency_file.write(f'    "stable_latency_avg": {np.average(stable_np_array)},\n')
            latency_file.write(f'    "stable_latency_p50": {np.percentile(stable_np_array, 50)},\n')
            latency_file.write(f'    "stable_latency_p90": {np.percentile(stable_np_array, 90)},\n')
            latency_file.write(f'    "stable_latency_p99": {np.percentile(stable_np_array, 99)},\n')
            latency_file.write(f'    "stable_latency_p100": {np.percentile(stable_np_array, 100)},\n')

            latency_file.write("}\n")


# =====================================================================
# ========== Extraction jobs to read tk.Variables from files ==========
# =====================================================================


class ExtractSearchErrorRateJob(Job):
    def __init__(self, search_error_file: tk.Path) -> None:
        self.search_error_file = search_error_file
        self.out_skipped = self.output_var("skipped")
        self.out_skipped_rate = self.output_var("skipped_rate")
        self.out_correct = self.output_var("correct")
        self.out_correct_rate = self.output_var("correct_rate")
        self.out_search_errors = self.output_var("search_errors")
        self.out_search_error_rate = self.output_var("search_error_rate")
        self.out_model_errors = self.output_var("model_errors")
        self.out_model_error_rate = self.output_var("model_error_rate")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.search_error_file.get(), "r") as f:
            result_dict = eval(f.read())
            self.out_skipped.set(result_dict["total_skipped"])
            self.out_skipped_rate.set(result_dict["skipped_rate"])
            self.out_correct.set(result_dict["total_correct"])
            self.out_correct_rate.set(result_dict["correct_rate"])
            self.out_search_errors.set(result_dict["total_search_errors"])
            self.out_search_error_rate.set(result_dict["search_error_rate"])
            self.out_model_errors.set(result_dict["total_model_errors"])
            self.out_model_error_rate.set(result_dict["model_error_rate"])


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


class ExtractSearchLatenciesJob(Job):
    def __init__(self, latency_file: tk.Path) -> None:
        self.latency_file = latency_file

        self.out_unstable_latency_avg = self.output_var("unstable_latency_avg")
        self.out_unstable_latency_p50 = self.output_var("unstable_latency_p50")
        self.out_unstable_latency_p90 = self.output_var("unstable_latency_p90")
        self.out_unstable_latency_p99 = self.output_var("unstable_latency_p99")
        self.out_unstable_latency_p100 = self.output_var("unstable_latency_p100")
        self.out_stable_latency_avg = self.output_var("stable_latency_avg")
        self.out_stable_latency_p50 = self.output_var("stable_latency_p50")
        self.out_stable_latency_p90 = self.output_var("stable_latency_p90")
        self.out_stable_latency_p99 = self.output_var("stable_latency_p99")
        self.out_stable_latency_p100 = self.output_var("stable_latency_p100")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.latency_file.get(), "r") as f:
            result_dict = eval(f.read())
            self.out_unstable_latency_avg.set(result_dict["unstable_latency_avg"])
            self.out_unstable_latency_p50.set(result_dict["unstable_latency_p50"])
            self.out_unstable_latency_p90.set(result_dict["unstable_latency_p90"])
            self.out_unstable_latency_p99.set(result_dict["unstable_latency_p99"])
            self.out_unstable_latency_p100.set(result_dict["unstable_latency_p100"])
            self.out_stable_latency_avg.set(result_dict["stable_latency_avg"])
            self.out_stable_latency_p50.set(result_dict["stable_latency_p50"])
            self.out_stable_latency_p90.set(result_dict["stable_latency_p90"])
            self.out_stable_latency_p99.set(result_dict["stable_latency_p99"])
            self.out_stable_latency_p100.set(result_dict["stable_latency_p100"])


class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    start_time: int
    end_time: int


# =============================
# ========== Helpers ==========
# =============================


def _samples_to_frames(n_samples: int, sample_rate: int, frame_shift_seconds: float) -> int:
    return int(np.round(n_samples / (sample_rate * frame_shift_seconds)))


def _frames_to_seconds(n_frames: int, frame_shift_seconds: float) -> float:
    return float(n_frames) * frame_shift_seconds


def _seconds_to_samples(n_seconds: float, sample_rate: int) -> int:
    return int(n_seconds * sample_rate)


def _samples_to_seconds(n_samples: int, sample_rate: int) -> float:
    return float(n_samples) / sample_rate


def _traceback_to_string(traceback: List[TracebackItem]) -> str:
    traceback_str = " ".join(item.lemma for item in traceback)
    traceback_str = traceback_str.replace("<s>", "")
    traceback_str = traceback_str.replace("</s>", "")
    traceback_str = traceback_str.replace("<blank>", "")
    traceback_str = traceback_str.replace("[BLANK] [1]", "")
    traceback_str = traceback_str.replace("[BLANK]", "")
    traceback_str = traceback_str.replace("<silence>", "")
    traceback_str = traceback_str.replace("[SILENCE]", "")
    traceback_str = traceback_str.replace("[SENTENCE-END]", "")
    traceback_str = " ".join(traceback_str.split())
    return traceback_str


def _traceback_to_transcription(traceback: List[TracebackItem]) -> str:
    traceback_str = _traceback_to_string(traceback)
    return traceback_str.replace("@@ ", "")


def _traceback_to_score(traceback: List[TracebackItem]) -> float:
    if len(traceback) > 0:
        return traceback[-1].am_score + traceback[-1].lm_score
    else:
        return float("inf")


# ==========================================
# ========== Forward step classes ==========
# ==========================================


class EncoderModel(Protocol):
    def forward(self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor) -> torch.Tensor: ...


class OfflineRasrRecogForwardStep:
    def __init__(self, rasr_config_file: tk.Path, sample_rate: int) -> None:
        from librasr import SearchAlgorithm

        self.rasr_config_file = rasr_config_file
        self.sample_rate = sample_rate
        self.search_algorithm: SearchAlgorithm = self.init_search_algorithm()

    def init_search_algorithm(self):
        from librasr import Configuration, SearchAlgorithm

        config = Configuration()
        config.set_from_file(self.rasr_config_file.get())

        return SearchAlgorithm(config=config)

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["search_algorithm"] = None

        return d

    def __setstate__(self, d) -> None:
        self.__dict__ = d
        self.search_algorithm = self.init_search_algorithm()

    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        raw_data = extern_data.as_raw_tensor_dict()
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        tokens_arrays = []
        token_lengths = []

        encoder_times = []
        search_times = []

        for b in range(audio_samples.size(0)):
            seq_samples_size = audio_samples_size[b : b + 1]
            seq_samples = audio_samples[b : b + 1, : seq_samples_size[0]]  # [1, T, 1]

            encoder_start = perf_counter()
            encoder_states = model.forward(seq_samples, seq_samples_size)
            encoder_time = perf_counter() - encoder_start
            encoder_times.append(encoder_time)

            encoder_states = encoder_states.to(device="cpu")

            search_start = perf_counter()

            traceback = self.search_algorithm.recognize_segment(features=encoder_states)
            search_time = perf_counter() - search_start
            search_times.append(search_time)

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)

            recog_str = _traceback_to_string(traceback)

            tokens_array = np.array(recog_str.split(), dtype="U")
            tokens_arrays.append(tokens_array)
            token_lengths.append(len(tokens_array))

            seq_time = _samples_to_seconds(seq_samples_size[0], self.sample_rate)

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

        enc_time_tensor = Tensor(
            name="enc_time",
            dtype="float32",
            raw_tensor=np.array(encoder_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(enc_time_tensor, name="enc_time")

        audio_samples_size_tensor = Tensor(
            name="audio_samples_size",
            dtype="int32",
            raw_tensor=audio_samples_size,
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(audio_samples_size_tensor, name="audio_samples_size")

        search_time_tensor = Tensor(
            name="search_time",
            dtype="float32",
            raw_tensor=np.array(search_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(search_time_tensor, name="search_time")


class OfflineRasrRecogForwardStepWithSearchErrors(OfflineRasrRecogForwardStep):
    def __init__(self, recog_rasr_config_file: tk.Path, align_rasr_config_file: tk.Path, sample_rate: int) -> None:
        super().__init__(rasr_config_file=recog_rasr_config_file, sample_rate=sample_rate)

        from librasr import Aligner

        self.align_rasr_config_file = align_rasr_config_file
        self.aligner: Aligner = self.init_aligner()

    def init_aligner(self):
        from librasr import Aligner, Configuration

        config = Configuration()
        config.set_from_file(self.align_rasr_config_file.get())

        return Aligner(config=config)

    def __getstate__(self) -> dict:
        result = super().__getstate__()
        result["align_rasr_config_file"] = self.align_rasr_config_file
        result["aligner"] = None

        return result

    def __setstate__(self, d) -> None:
        super().__setstate__(d)
        self.aligner = self.init_aligner()

    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        raw_data = extern_data.as_raw_tensor_dict()
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        tokens_arrays = []
        token_lengths = []

        search_errors = []
        model_errors = []
        skipped = []
        correct = []

        encoder_times = []
        search_times = []

        for b in range(audio_samples.size(0)):
            seq_samples_size = audio_samples_size[b : b + 1]
            seq_samples = audio_samples[b : b + 1, : seq_samples_size[0]]  # [1, T, 1]

            encoder_start = perf_counter()
            encoder_states = model.forward(seq_samples, seq_samples_size)
            encoder_time = perf_counter() - encoder_start
            encoder_times.append(encoder_time)

            encoder_states = encoder_states.to(device="cpu")

            search_start = perf_counter()
            traceback = self.search_algorithm.recognize_segment(features=encoder_states)
            search_time = perf_counter() - search_start
            search_times.append(search_time)

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)

            recog_str = _traceback_to_string(traceback)

            tokens_array = np.array(recog_str.split(), dtype="U")
            tokens_arrays.append(tokens_array)
            token_lengths.append(len(tokens_array))

            seq_time = _samples_to_seconds(seq_samples_size[0], self.sample_rate)

            align_traceback = self.aligner.align_segment(
                features=encoder_states, orth=orths[b] + " "
            )  # RASR requires a trailing space in transcription

            recog_transcription = _traceback_to_transcription(traceback)
            align_transcription = _traceback_to_transcription(align_traceback)

            recog_score = _traceback_to_score(traceback)
            align_score = _traceback_to_score(align_traceback)

            if align_transcription != orths[b]:
                print("    Could not successfully compute forced alignment. Transcription may contain OOV words.")
                skipped.append(1)
                search_errors.append(0)
                model_errors.append(0)
                correct.append(0)
            elif recog_transcription == orths[b]:
                print("    Correct transcription found.")
                skipped.append(0)
                search_errors.append(0)
                model_errors.append(0)
                correct.append(1)
            elif align_score < recog_score:
                print(
                    f"    Encountered search error. Forced alignment has score {align_score} while search has score {recog_score}"
                )
                skipped.append(0)
                search_errors.append(1)
                model_errors.append(0)
                correct.append(0)
            else:
                print(
                    f"    Encountered model error. Forced alignment has score {align_score} while search has score {recog_score}"
                )
                skipped.append(0)
                search_errors.append(0)
                model_errors.append(1)
                correct.append(0)

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

        search_errors_tensor = Tensor(
            name="search_errors",
            dtype="int32",
            raw_tensor=np.array(search_errors, dtype=np.int32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        model_errors_tensor = Tensor(
            name="model_errors",
            dtype="int32",
            raw_tensor=np.array(model_errors, dtype=np.int32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        skipped_tensor = Tensor(
            name="skipped",
            dtype="int32",
            raw_tensor=np.array(skipped, dtype=np.int32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        correct_tensor = Tensor(
            name="correct",
            dtype="int32",
            raw_tensor=np.array(correct, dtype=np.int32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext.raw_tensor = tokens_len_array
        run_ctx.mark_as_output(tokens_tensor, name="tokens")

        run_ctx.mark_as_output(search_errors_tensor, name="search_errors")
        run_ctx.mark_as_output(model_errors_tensor, name="model_errors")
        run_ctx.mark_as_output(skipped_tensor, name="skipped")
        run_ctx.mark_as_output(correct_tensor, name="correct")

        enc_time_tensor = Tensor(
            name="enc_time",
            dtype="float32",
            raw_tensor=np.array(encoder_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(enc_time_tensor, name="enc_time")

        audio_samples_size_tensor = Tensor(
            name="audio_samples_size",
            dtype="int32",
            raw_tensor=audio_samples_size,
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(audio_samples_size_tensor, name="audio_samples_size")

        search_time_tensor = Tensor(
            name="search_time",
            dtype="float32",
            raw_tensor=np.array(search_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(search_time_tensor, name="search_time")


class StreamingRasrRecogForwardStep:
    def __init__(
        self,
        rasr_config_file: tk.Path,
        sample_rate: int,
        encoder_frame_shift_seconds: float,
        chunk_history_seconds: float,
        chunk_center_seconds: float,
        chunk_future_seconds: float,
    ) -> None:
        from librasr import SearchAlgorithm

        self.rasr_config_file = rasr_config_file

        self.search_algorithm: SearchAlgorithm = self.init_search_algorithm()

        self.sample_rate = sample_rate

        self.chunk_history_samples = _seconds_to_samples(chunk_history_seconds, sample_rate)
        self.chunk_center_samples = _seconds_to_samples(chunk_center_seconds, sample_rate)
        self.chunk_future_samples = _seconds_to_samples(chunk_future_seconds, sample_rate)

        self.encoder_frame_shift_seconds = encoder_frame_shift_seconds

    def init_search_algorithm(self):
        from librasr import Configuration, SearchAlgorithm

        config = Configuration()
        config.set_from_file(self.rasr_config_file.get())

        return SearchAlgorithm(config=config)

    def __getstate__(self) -> dict:
        result = dict(self.__dict__)
        result["search_algorithm"] = None
        return result

    def __setstate__(self, d) -> None:
        self.__dict__ = d
        self.search_algorithm = self.init_search_algorithm()

    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        raw_data = extern_data.as_raw_tensor_dict()
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        tokens_arrays = []
        token_lengths = []

        unstable_latency_arrays = []
        unstable_latency_lengths = []
        stable_latency_arrays = []
        stable_latency_lengths = []

        for b in range(audio_samples.size(0)):
            self.search_algorithm.reset()
            self.search_algorithm.enter_segment()

            seq_samples_size = audio_samples_size[b]
            seq_samples = audio_samples[b : b + 1, :seq_samples_size]  # [1, T, 1]

            chunk_center_start_sample = 0
            num_prev_stable_results = 0
            prev_chunk_end_sample = 0

            leftover_processing_time = 0

            unstable_latencies = []
            stable_latencies = []

            while chunk_center_start_sample < seq_samples_size:
                chunk_center_end_sample = min(seq_samples_size, chunk_center_start_sample + self.chunk_center_samples)

                chunk_start_sample = max(0, chunk_center_start_sample - self.chunk_history_samples)
                chunk_end_sample = min(seq_samples_size, chunk_center_end_sample + self.chunk_future_samples)

                # If encoder + search processing time is shorter than the new audio, it gets absorbed. Otherwise, the next
                # processing step has to wait a bit even after all audio is available until the previous processing step is complete.
                leftover_processing_time = max(
                    0,
                    leftover_processing_time
                    - _samples_to_seconds(chunk_end_sample - prev_chunk_end_sample, self.sample_rate),
                )
                prev_chunk_end_sample = chunk_end_sample

                encoder_start = perf_counter()

                encoder_states = model.forward(
                    seq_samples[:, chunk_start_sample:chunk_end_sample],
                    torch.tensor([chunk_end_sample - chunk_start_sample], device=audio_samples_size.device),
                )

                encoder_elapsed = perf_counter() - encoder_start

                total_encoder_frames = encoder_states.size(1)

                center_start_frame = max(
                    0,
                    _samples_to_frames(
                        chunk_center_start_sample - chunk_start_sample,
                        self.sample_rate,
                        self.encoder_frame_shift_seconds,
                    ),
                )
                center_end_frame = min(
                    _samples_to_frames(
                        chunk_center_end_sample - chunk_start_sample,
                        self.sample_rate,
                        self.encoder_frame_shift_seconds,
                    ),
                    total_encoder_frames,
                )

                if center_end_frame > center_start_frame:
                    search_start = perf_counter()

                    self.search_algorithm.put_features(encoder_states[:, center_start_frame:center_end_frame])
                    full_traceback = self.search_algorithm.get_current_best_traceback()
                    stable_traceback = self.search_algorithm.get_current_stable_traceback()

                    search_elapsed = perf_counter() - search_start

                    # Realtime where we would be in the stream after sending all samples for current chunk, encoding them and running search
                    processing_finish_time = (
                        _samples_to_seconds(chunk_end_sample, self.sample_rate)
                        + leftover_processing_time
                        + encoder_elapsed
                        + search_elapsed
                    )

                    leftover_processing_time += encoder_elapsed + search_elapsed

                    new_unstable_results = full_traceback[len(stable_traceback) :]
                    new_stable_results = stable_traceback[num_prev_stable_results:]

                    # Count differences from when words in hypothesis have ended to where we are now
                    for stable_result in new_stable_results:
                        stable_latencies.append(
                            processing_finish_time
                            - _frames_to_seconds(stable_result.end_time, self.encoder_frame_shift_seconds)
                        )

                    for unstable_result in new_unstable_results:
                        unstable_latencies.append(
                            processing_finish_time
                            - _frames_to_seconds(unstable_result.end_time, self.encoder_frame_shift_seconds)
                        )

                    print("Intermediate hypothesis after processing chunk:")
                    print(
                        f'  Previous stable hypothesis: "{_traceback_to_string(stable_traceback[:num_prev_stable_results])}"'
                    )
                    print(f'  New stable extension: "{_traceback_to_string(new_stable_results)}"')
                    print(f'  New unstable extension: "{_traceback_to_string(new_unstable_results)}"')

                    num_prev_stable_results = len(stable_traceback)

                chunk_center_start_sample = chunk_center_end_sample

            self.search_algorithm.finish_segment()
            final_traceback = self.search_algorithm.get_current_best_traceback()

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)
            recog_str = _traceback_to_string(final_traceback)

            tokens_array = np.array(recog_str.split(), dtype="U")
            tokens_arrays.append(tokens_array)
            token_lengths.append(len(tokens_array))

            print(f"    Tokens: {tokens_array}")
            print()

            unstable_latency_arrays.append(np.array(unstable_latencies, dtype=np.float32))
            stable_latency_arrays.append(np.array(stable_latencies, dtype=np.float32))
            unstable_latency_lengths.append(len(unstable_latencies))
            stable_latency_lengths.append(len(stable_latencies))

        tokens_arrays_padded = [
            np.pad(tokens_array, pad_width=(0, np.max(token_lengths) - len(tokens_array)))
            for tokens_array in tokens_arrays
        ]

        tokens_tensor = Tensor(
            name="tokens", dtype="string", raw_tensor=np.stack(tokens_arrays_padded, axis=0), feature_dim_axis=None
        )
        tokens_len_array = np.array(token_lengths, dtype=np.int32)

        unstable_latency_arrays_padded = [
            np.pad(latency_array, pad_width=(0, np.max(unstable_latency_lengths) - len(latency_array)))
            for latency_array in unstable_latency_arrays
        ]
        unstable_latencies_tensor = Tensor(
            name="unstable_latencies",
            dtype="float32",
            raw_tensor=np.stack(unstable_latency_arrays_padded, axis=0),
            feature_dim_axis=None,
        )
        unstable_latency_lengths_array = np.array(unstable_latency_lengths, dtype=np.int32)

        stable_latency_arrays_padded = [
            np.pad(latency_array, pad_width=(0, np.max(stable_latency_lengths) - len(latency_array)))
            for latency_array in stable_latency_arrays
        ]
        stable_latencies_tensor = Tensor(
            name="stable_latencies",
            dtype="float32",
            raw_tensor=np.stack(stable_latency_arrays_padded, axis=0),
            feature_dim_axis=None,
        )
        stable_latency_lengths_array = np.array(stable_latency_lengths, dtype=np.int32)

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["tokens"].dims[1].dyn_size_ext.raw_tensor = tokens_len_array

            assert run_ctx.expected_outputs["unstable_latencies"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["unstable_latencies"].dims[
                1
            ].dyn_size_ext.raw_tensor = unstable_latency_lengths_array

            assert run_ctx.expected_outputs["stable_latencies"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["stable_latencies"].dims[1].dyn_size_ext.raw_tensor = stable_latency_lengths_array

        run_ctx.mark_as_output(tokens_tensor, name="tokens")
        run_ctx.mark_as_output(unstable_latencies_tensor, name="unstable_latencies")
        run_ctx.mark_as_output(stable_latencies_tensor, name="stable_latencies")


def recog_rasr_offline(
    descriptor: str,
    checkpoint: PtCheckpoint,
    rasr_config_file: tk.Path,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    encoder_serializers: Collection,
    sample_rate: int,
    gpu_mem_rqmt: int = 0,
) -> OfflineRecogResult:
    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "raw": {"feature_dim_axis": None, "time_dim_axis": None, "dtype": "string"},
            },
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                },
                "audio_samples_size": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "enc_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "search_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
            },
            "backend": "torch",
            "batch_size": 36_000 * 160,
        },
        python_prolog=recipe_imports + [ExternalImport(rasr_binary_path)],
        python_epilog=[
            encoder_serializers,
            Import(
                f"{OfflineSearchCallback.__module__}.{OfflineSearchCallback.__name__}",
                import_as="forward_callback",
            ),
            Import(
                f"{OfflineRasrRecogForwardStep.__module__}.{OfflineRasrRecogForwardStep.__name__}",
            ),
            Import("sisyphus.tk"),
            Call(
                OfflineRasrRecogForwardStep.__name__,
                kwargs=[
                    ("rasr_config_file", DelayedFormat('tk.Path("{}")', rasr_config_file)),
                    ("sample_rate", sample_rate),
                ],
                return_assign_variables="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

    output_files = ["search_out.py", "rtf.py", "rasr.recog.log"]

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device="gpu" if gpu_mem_rqmt > 0 else "cpu",
        mem_rqmt=16,
        time_rqmt=168,
    )
    recog_job.add_alias(f"recognition/{recog_corpus.corpus_name}/{descriptor}")
    if gpu_mem_rqmt > 0:
        recog_job.rqmt["gpu_mem"] = gpu_mem_rqmt

    for output_file in output_files:
        tk.register_output(
            f"recognition/{recog_corpus.corpus_name}/{descriptor}/{output_file}",
            recog_job.out_files[output_file],
        )

    extract_rtf_job = ExtractSearchRTFJob(rtf_file=recog_job.out_files["rtf.py"])
    tk.register_output(f"recognition/{recog_corpus.corpus_name}/{descriptor}/rtf", extract_rtf_job.out_total_rtf)

    word_file = SearchBPEtoWordsJob(recog_job.out_files["search_out.py"]).out_word_search_results

    recog_corpus_file: tk.Path = recog_corpus.bliss_corpus_file

    ctm_file = SearchWordsToCTMJob(word_file, bliss_corpus=recog_corpus_file).out_ctm_file
    score_job = recog_corpus.score_ctm(ctm_file)
    tk.register_output(f"recognition/{recog_corpus.corpus_name}/{descriptor}/scoring_reports", score_job.out_report_dir)

    return OfflineRecogResult(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        wer=score_job.out_wer,
        deletion=score_job.out_percent_deletions,
        insertion=score_job.out_percent_insertions,
        substitution=score_job.out_percent_substitution,
        enc_rtf=extract_rtf_job.out_enc_rtf,
        search_rtf=extract_rtf_job.out_search_rtf,
        total_rtf=extract_rtf_job.out_total_rtf,
    )


def recog_rasr_offline_with_search_errors(
    descriptor: str,
    checkpoint: PtCheckpoint,
    recog_rasr_config_file: tk.Path,
    align_rasr_config_file: tk.Path,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    encoder_serializers: Collection,
    sample_rate: int,
    gpu_mem_rqmt: int = 0,
) -> OfflineRecogResultWithSearchErrors:
    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "raw": {"feature_dim_axis": None, "time_dim_axis": None, "dtype": "string"},
            },
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                },
                "search_errors": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "model_errors": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "correct": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "skipped": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "audio_samples_size": {
                    "dtype": "int32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "enc_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
                "search_time": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
                },
            },
            "backend": "torch",
            "batch_size": 36_000 * 160,
        },
        python_prolog=recipe_imports + [ExternalImport(rasr_binary_path)],
        python_epilog=[
            encoder_serializers,
            Import(
                f"{OfflineSearchCallbackWithSearchErrors.__module__}.{OfflineSearchCallbackWithSearchErrors.__name__}",
                import_as="forward_callback",
            ),
            Import(
                f"{OfflineRasrRecogForwardStepWithSearchErrors.__module__}.{OfflineRasrRecogForwardStepWithSearchErrors.__name__}",
            ),
            Import("sisyphus.tk"),
            Call(
                OfflineRasrRecogForwardStepWithSearchErrors.__name__,
                kwargs=[
                    ("recog_rasr_config_file", DelayedFormat('tk.Path("{}")', recog_rasr_config_file)),
                    ("align_rasr_config_file", DelayedFormat('tk.Path("{}")', align_rasr_config_file)),
                    ("sample_rate", sample_rate),
                ],
                return_assign_variables="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

    output_files = ["search_out.py", "rtf.py", "search_errors.py", "rasr.recog.log", "rasr.align.log"]

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device="gpu" if gpu_mem_rqmt > 0 else "cpu",
        mem_rqmt=16,
        time_rqmt=168,
    )
    recog_job.add_alias(f"recognition/{recog_corpus.corpus_name}/{descriptor}")
    if gpu_mem_rqmt > 0:
        recog_job.rqmt["gpu_mem"] = gpu_mem_rqmt

    for output_file in output_files:
        tk.register_output(
            f"recognition/{recog_corpus.corpus_name}/{descriptor}/{output_file}",
            recog_job.out_files[output_file],
        )

    extract_search_error_job = ExtractSearchErrorRateJob(search_error_file=recog_job.out_files["search_errors.py"])
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/search_error_rate",
        extract_search_error_job.out_search_error_rate,
    )

    extract_rtf_job = ExtractSearchRTFJob(rtf_file=recog_job.out_files["rtf.py"])
    tk.register_output(f"recognition/{recog_corpus.corpus_name}/{descriptor}/rtf", extract_rtf_job.out_total_rtf)

    word_file = SearchBPEtoWordsJob(recog_job.out_files["search_out.py"]).out_word_search_results

    recog_corpus_file: tk.Path = recog_corpus.bliss_corpus_file

    ctm_file = SearchWordsToCTMJob(word_file, bliss_corpus=recog_corpus_file).out_ctm_file
    score_job = recog_corpus.score_ctm(ctm_file)
    tk.register_output(f"recognition/{recog_corpus.corpus_name}/{descriptor}/scoring_reports", score_job.out_report_dir)

    return OfflineRecogResultWithSearchErrors(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        wer=score_job.out_wer,
        search_error_rate=extract_search_error_job.out_search_error_rate,
        model_error_rate=extract_search_error_job.out_model_error_rate,
        skipped_rate=extract_search_error_job.out_skipped_rate,
        correct_rate=extract_search_error_job.out_correct_rate,
        deletion=score_job.out_percent_deletions,
        insertion=score_job.out_percent_insertions,
        substitution=score_job.out_percent_substitution,
        enc_rtf=extract_rtf_job.out_enc_rtf,
        search_rtf=extract_rtf_job.out_search_rtf,
        total_rtf=extract_rtf_job.out_total_rtf,
    )


def recog_rasr_streaming(
    descriptor: str,
    checkpoint: PtCheckpoint,
    rasr_config_file: tk.Path,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    encoder_serializers: Collection,
    encoder_frame_shift_seconds: float,
    chunk_history_seconds: float,
    chunk_center_seconds: float,
    chunk_future_seconds: float,
    sample_rate: int,
    gpu_mem_rqmt: int = 0,
) -> StreamingRecogResult:
    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "raw": {"feature_dim_axis": None, "time_dim_axis": None, "dtype": "string"},
            },
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                },
                "unstable_latencies": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                },
                "stable_latencies": {
                    "dtype": "float32",
                    "feature_dim_axis": None,
                },
            },
            "backend": "torch",
            "batch_size": 36_000 * 160,
        },
        python_prolog=recipe_imports + [ExternalImport(rasr_binary_path)],
        python_epilog=[
            encoder_serializers,
            Import(
                f"{StreamingSearchCallback.__module__}.{StreamingSearchCallback.__name__}",
                import_as="forward_callback",
            ),
            Import(
                f"{StreamingRasrRecogForwardStep.__module__}.{StreamingRasrRecogForwardStep.__name__}",
            ),
            Import("sisyphus.tk"),
            Call(
                StreamingRasrRecogForwardStep.__name__,
                kwargs=[
                    ("rasr_config_file", DelayedFormat('tk.Path("{}")', rasr_config_file)),
                    ("sample_rate", sample_rate),
                    ("encoder_frame_shift_seconds", encoder_frame_shift_seconds),
                    ("chunk_history_seconds", chunk_history_seconds),
                    ("chunk_center_seconds", chunk_center_seconds),
                    ("chunk_future_seconds", chunk_future_seconds),
                ],
                return_assign_variables="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

    output_files = ["search_out.py", "latencies.py", "rasr.recog.log"]

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device="gpu" if gpu_mem_rqmt > 0 else "cpu",
        mem_rqmt=16,
        time_rqmt=168,
    )
    recog_job.add_alias(f"recognition/{recog_corpus.corpus_name}/{descriptor}")
    if gpu_mem_rqmt > 0:
        recog_job.rqmt["gpu_mem"] = gpu_mem_rqmt

    for output_file in output_files:
        tk.register_output(
            f"recognition/{recog_corpus.corpus_name}/{descriptor}/{output_file}",
            recog_job.out_files[output_file],
        )

    extract_latencies_job = ExtractSearchLatenciesJob(latency_file=recog_job.out_files["latencies.py"])
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/unstable_latency_avg",
        extract_latencies_job.out_unstable_latency_avg,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/unstable_latency_p50",
        extract_latencies_job.out_unstable_latency_p50,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/unstable_latency_p90",
        extract_latencies_job.out_unstable_latency_p90,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/unstable_latency_p99",
        extract_latencies_job.out_unstable_latency_p99,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/unstable_latency_p100",
        extract_latencies_job.out_unstable_latency_p100,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/stable_latency_avg",
        extract_latencies_job.out_stable_latency_avg,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/stable_latency_p50",
        extract_latencies_job.out_stable_latency_p50,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/stable_latency_p90",
        extract_latencies_job.out_stable_latency_p90,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/stable_latency_p99",
        extract_latencies_job.out_stable_latency_p99,
    )
    tk.register_output(
        f"recognition/{recog_corpus.corpus_name}/{descriptor}/latencies/stable_latency_p100",
        extract_latencies_job.out_stable_latency_p100,
    )

    word_file = SearchBPEtoWordsJob(recog_job.out_files["search_out.py"]).out_word_search_results

    recog_corpus_file: tk.Path = recog_corpus.bliss_corpus_file

    ctm_file = SearchWordsToCTMJob(word_file, bliss_corpus=recog_corpus_file).out_ctm_file
    score_job = recog_corpus.score_ctm(ctm_file)
    tk.register_output(f"recognition/{recog_corpus.corpus_name}/{descriptor}/scoring_reports", score_job.out_report_dir)

    return StreamingRecogResult(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        wer=score_job.out_wer,
        deletion=score_job.out_percent_deletions,
        insertion=score_job.out_percent_insertions,
        substitution=score_job.out_percent_substitution,
        unstable_latency_avg=extract_latencies_job.out_unstable_latency_avg,
        unstable_latency_p50=extract_latencies_job.out_unstable_latency_p50,
        unstable_latency_p90=extract_latencies_job.out_unstable_latency_p90,
        unstable_latency_p99=extract_latencies_job.out_unstable_latency_p99,
        unstable_latency_p100=extract_latencies_job.out_unstable_latency_p100,
        stable_latency_avg=extract_latencies_job.out_stable_latency_avg,
        stable_latency_p50=extract_latencies_job.out_stable_latency_p50,
        stable_latency_p90=extract_latencies_job.out_stable_latency_p90,
        stable_latency_p99=extract_latencies_job.out_stable_latency_p99,
        stable_latency_p100=extract_latencies_job.out_stable_latency_p100,
    )
