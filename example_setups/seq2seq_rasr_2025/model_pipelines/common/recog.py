__all__ = [
    "RecogResult",
    "OfflineRecogParameters",
    "StreamingRecogParameters",
    "recog_rasr_offline",
    "recog_rasr_streaming",
]

import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, List, Optional, Protocol

import numpy as np
import torch
from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.util import DelayedFormat
from i6_experiments.common.setups.serialization import Call, Collection, ExternalImport, Import
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import Tensor, TensorDict
from sisyphus import Job, Task, tk

from ...data.base import DataConfig
from ...tools import rasr_binary_path, returnn_python_exe, returnn_root
from .corpus import ScorableCorpus
from .serializers import recipe_imports

# =============================
# ======== Dataclasses ========
# =============================


@dataclass
class RecogResult:
    descriptor: str
    corpus_name: str
    wer: Optional[tk.Variable]
    deletion: Optional[tk.Variable]
    insertion: Optional[tk.Variable]
    substitution: Optional[tk.Variable]
    search_error_rate: Optional[tk.Variable]
    model_error_rate: Optional[tk.Variable]
    correct_rate: Optional[tk.Variable]
    skipped_rate: Optional[tk.Variable]
    enc_rtf: Optional[tk.Variable]
    search_rtf: Optional[tk.Variable]
    total_rtf: Optional[tk.Variable]
    unstable_latency_stats: Optional[tk.Variable]
    stable_latency_stats: Optional[tk.Variable]
    step_hyps_stats: Optional[tk.Variable]
    step_word_end_hyps_stats: Optional[tk.Variable]
    step_trees_stats: Optional[tk.Variable]


@dataclass
class OfflineRecogParameters:
    mem_rqmt: int = 16
    gpu_mem_rqmt: int = 0


@dataclass
class StreamingRecogParameters:
    encoder_frame_shift_seconds: float
    chunk_history_seconds: float = 10.0
    chunk_center_seconds: float = 1.0
    chunk_future_seconds: float = 1.0
    mem_rqmt: int = 16
    gpu_mem_rqmt: int = 0


# =============================
# ========== Helpers ==========
# =============================


class ExtractSearchErrorDataJob(Job):
    def __init__(self, search_errors_file: tk.Path) -> None:
        self.search_errors_file = search_errors_file
        self.out_total_skipped = self.output_var("total_skipped")
        self.out_skipped_rate = self.output_var("skipped_rate")
        self.out_total_correct = self.output_var("total_correct")
        self.out_correct_rate = self.output_var("correct_rate")
        self.out_total_search_errors = self.output_var("total_search_errors")
        self.out_search_error_rate = self.output_var("search_error_rate")
        self.out_total_model_errors = self.output_var("total_model_errors")
        self.out_model_error_rate = self.output_var("model_error_rate")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.search_errors_file, "rt") as f:
            d = eval(f.read())

        self.out_total_skipped.set(d["total_skipped"])
        self.out_skipped_rate.set(d["skipped_rate"])
        self.out_total_correct.set(d["total_correct"])
        self.out_correct_rate.set(d["correct_rate"])
        self.out_total_search_errors.set(d["total_search_errors"])
        self.out_search_error_rate.set(d["search_error_rate"])
        self.out_total_model_errors.set(d["total_model_errors"])
        self.out_model_error_rate.set(d["model_error_rate"])


class ExtractRTFDataJob(Job):
    def __init__(self, rtf_file: tk.Path) -> None:
        self.rtf_file = rtf_file
        self.out_enc_rtf = self.output_var("enc_rtf")
        self.out_search_rtf = self.output_var("search_rtf")
        self.out_total_rtf = self.output_var("total_rtf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.rtf_file, "rt") as f:
            d = eval(f.read())

        self.out_enc_rtf.set(d["enc_rtf"])
        self.out_search_rtf.set(d["search_rtf"])
        self.out_total_rtf.set(d["total_rtf"])


class ExtractStatisticsJob(Job):
    def __init__(self, stat_file: tk.Path) -> None:
        self.stat_file = stat_file
        self.out_stats = self.output_var(
            "statistics", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        with open(self.stat_file, "rt") as f:
            d = eval(f.read())
        self.out_stats.set(d)


class ExtractRasrStatisticsJob(Job):
    def __init__(self, rasr_log_file: tk.Path) -> None:
        self.rasr_log_file = rasr_log_file

        self.out_step_hyps = self.output_var(
            "step_hyps", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )
        self.out_step_word_end_hyps = self.output_var(
            "step_word_end_hyps", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )
        self.out_step_trees = self.output_var(
            "step_trees", backup={"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "p100": 0.0}
        )

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        import re

        step_hyps_counts = []
        step_word_end_hyps_counts = []
        step_trees_counts = []

        with open(self.rasr_log_file.get(), "r") as f:
            for line in f:
                for log_str, counter in [
                    ("<num-hyps-after-beam-pruning>", step_hyps_counts),
                    ("<num-word-end-hyps-after-beam-pruning>", step_word_end_hyps_counts),
                    ("<num-active-trees>", step_trees_counts),
                ]:
                    if log_str in line:
                        value = int(re.split(r"[<>]+", line)[2].strip())
                        counter.append(value)

        self.out_step_hyps.set(statistics_from_data(step_hyps_counts))
        self.out_step_word_end_hyps.set(statistics_from_data(step_word_end_hyps_counts))
        self.out_step_trees.set(statistics_from_data(step_trees_counts))


class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    confidence_score: Optional[float]
    start_time: int
    end_time: int


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


def _traceback_to_ctm_str(traceback: List[TracebackItem], ms_per_frame: int) -> str:
    lines = []
    start_frame: Optional[int] = None
    current_word = ""
    confidences = []
    for item in traceback:
        if item.confidence_score is not None:
            for _ in range(item.start_time, item.end_time):
                confidences.append(item.confidence_score)
        if item.lemma.startswith("[") and item.lemma.endswith("]"):
            continue
        if item.lemma.startswith("<") and item.lemma.endswith(">"):
            continue
        if start_frame is None:
            start_frame = item.start_time
        current_word += item.lemma.replace("@@", "")
        if item.lemma.endswith("@@"):
            continue
        duration_frames = item.end_time - start_frame

        start_time = start_frame * 0.001 * ms_per_frame
        duration_time = duration_frames * 0.001 * ms_per_frame
        if len(confidences) == 0:
            avg_confidence = 0.99
        else:
            avg_confidence = sum(confidences) / len(confidences)
        lines.append(f"[REC_NAME] 1 {start_time:.3f} {duration_time:.3f} {current_word} {avg_confidence:.2f}")
        start_frame = None
        current_word = ""
        confidences.clear()

    if len(lines) == 0:
        return "[REC_NAME] 1 0.000 0.010 <empty-sequence> 0.99\n"
    else:
        return "\n".join(lines) + "\n"


def _traceback_to_score(traceback: List[TracebackItem]) -> float:
    if len(traceback) > 0:
        return traceback[-1].am_score + traceback[-1].lm_score
    else:
        return float("inf")


def statistics_from_data(data: list[float]) -> dict[str, float]:
    if len(data) == 0:
        avg = 0.0
        p50 = 0.0
        p90 = 0.0
        p99 = 0.0
        p100 = 0.0
    else:
        data_array = np.array(data)
        avg = np.average(data_array).astype(float)
        p50 = np.percentile(data_array, 50).astype(float)
        p90 = np.percentile(data_array, 90).astype(float)
        p99 = np.percentile(data_array, 99).astype(float)
        p100 = np.percentile(data_array, 100).astype(float)

    return {
        "avg": avg,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "p100": p100,
    }


# =======================================================
# ========== Callbacks for result file writing ==========
# =======================================================


class SearchCallback(ForwardCallbackIface):
    def init(self, *, model):
        self.ctm_file = open("search_out.ctm", "w")
        self.ctm_file.write(";; <name> <track> <start> <duration> <word> <confidence>\n")

        self.total_skipped = 0
        self.total_correct = 0
        self.total_search_errors = 0
        self.total_model_errors = 0

        self.total_seqs = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        self.total_seqs += 1

        raw_outputs = outputs.as_raw_tensor_dict()
        ctm_str = raw_outputs.get("ctm_string")
        assert ctm_str is not None
        ctm_str = ctm_str.item()
        rec_name = seq_tag.split("/")[1]
        self.ctm_file.write(f";; {seq_tag}\n")
        self.ctm_file.write(ctm_str.replace("[REC_NAME]", rec_name))

        self.total_skipped += raw_outputs.get("skipped", 0)
        self.total_correct += raw_outputs.get("correct", 0)
        self.total_search_errors += raw_outputs.get("search_errors", 0)
        self.total_model_errors += raw_outputs.get("model_errors", 0)

    def finish(self):
        self.ctm_file.close()

        if self.total_skipped + self.total_correct + self.total_search_errors + self.total_model_errors > 0:
            search_error_data = {
                "total_skipped": self.total_skipped,
                "skipped_rate": self.total_skipped / self.total_seqs,
                "total_correct": self.total_correct,
                "correct_rate": self.total_correct / self.total_seqs,
                "total_search_errors": self.total_search_errors,
                "search_error_rate": self.total_search_errors / self.total_seqs,
                "total_model_errors": self.total_model_errors,
                "model_error_rate": self.total_model_errors / self.total_seqs,
            }
            print()
            print(
                f"Total search errors: {search_error_data['total_search_errors']} ({search_error_data['search_error_rate']:.2f}%)"
            )
            print(
                f"Total model errors: {search_error_data['total_model_errors']} ({search_error_data['model_error_rate']:.2f}%)"
            )
            print(
                f"Total correct seqs: {search_error_data['total_correct']} ({search_error_data['correct_rate']:.2f}%)"
            )
            print(f"Total skipped: {search_error_data['total_skipped']} ({search_error_data['skipped_rate']:.2f}%)")
            print()

            with open("search_errors.py", "w") as search_error_file:
                search_error_file.write(pprint.pformat(search_error_data))


class OfflineSearchCallback(SearchCallback):
    def init(self, *, model):
        super().init(model=model)

        self.total_audio_time = 0
        self.total_enc_time = 0
        self.total_search_time = 0

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)
        raw_outputs = outputs.as_raw_tensor_dict()

        self.total_audio_time += raw_outputs.get("audio_time", 0)
        self.total_enc_time += raw_outputs.get("enc_time", 0)
        self.total_search_time += raw_outputs.get("search_time", 0)

    def finish(self):
        super().finish()

        rtf_data = {
            "audio_seconds": self.total_audio_time,
            "enc_seconds": self.total_enc_time,
            "enc_rtf": self.total_enc_time / self.total_audio_time,
            "enc_rtfx": self.total_audio_time / self.total_enc_time,
            "search_seconds": self.total_search_time,
            "search_rtf": self.total_search_time / self.total_audio_time,
            "search_rtfx": self.total_audio_time / self.total_search_time,
            "total_seconds": self.total_enc_time + self.total_search_time,
            "total_rtf": (self.total_enc_time + self.total_search_time) / self.total_audio_time,
            "total_rtfx": self.total_audio_time / (self.total_enc_time + self.total_search_time),
        }

        print(
            f"Total encoder time: {self.total_enc_time:.2f} seconds, AM-RTF: {rtf_data['enc_rtf']}, XRTF: {rtf_data['enc_rtfx']}"
        )
        print(
            f"Total search time: {self.total_search_time:.2f} seconds, search-RTF: {rtf_data['search_rtf']}, RTFX: {rtf_data['search_rtfx']}"
        )
        print(
            f"Total time: {rtf_data['total_seconds']:.2f} seconds, RTF: {rtf_data['total_rtf']}, RTFX: {rtf_data['total_rtfx']}"
        )
        print()

        with open("rtf.py", "w") as rtf_file:
            rtf_file.write(pprint.pformat(rtf_data))


class StreamingSearchCallback(SearchCallback):
    def init(self, *, model):
        super().init(model=model)

        self.unstable_latencies = []
        self.stable_latencies = []

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        super().process_seq(seq_tag=seq_tag, outputs=outputs)

        raw_outputs = outputs.as_raw_tensor_dict()

        self.unstable_latencies.extend(raw_outputs.get("unstable_latencies", []))
        self.stable_latencies.extend(raw_outputs.get("stable_latencies", []))

    def finish(self):
        super().finish()

        unstable_latency_stats = statistics_from_data(self.unstable_latencies)
        stable_latency_stats = statistics_from_data(self.stable_latencies)

        print(f"Unstable latencies: {', '.join(f'{key}={val:.2f}' for key, val in unstable_latency_stats.items())}")
        print(f"Stable latencies: {', '.join(f'{key}={val:.2f}' for key, val in stable_latency_stats.items())}")

        with open("unstable_latencies.py", "w") as latency_file:
            latency_file.write(pprint.pformat(unstable_latency_stats))

        with open("stable_latencies.py", "w") as latency_file:
            latency_file.write(pprint.pformat(stable_latency_stats))


# ==========================================
# ========== Forward step classes ==========
# ==========================================


class EncoderModel(Protocol):
    def forward(self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor) -> torch.Tensor: ...


class RasrRecogForwardStep(ABC):
    def __init__(
        self,
        recog_rasr_config_file: tk.Path,
        align_rasr_config_file: Optional[tk.Path] = None,
        sample_rate: int = 16000,
    ) -> None:
        self.recog_rasr_config_file = recog_rasr_config_file
        self.sample_rate = sample_rate
        self.search_algorithm = self.init_search_algorithm()

        self.align_rasr_config_file = align_rasr_config_file
        self.aligner = self.init_aligner()

    def init_search_algorithm(self):
        if self.recog_rasr_config_file is None:
            return None

        from librasr import Configuration, SearchAlgorithm

        config = Configuration()
        config.set_from_file(self.recog_rasr_config_file.get())

        return SearchAlgorithm(config=config)

    def init_aligner(self):
        if self.align_rasr_config_file is None:
            return None

        from librasr import Aligner, Configuration

        config = Configuration()
        config.set_from_file(self.align_rasr_config_file.get())

        return Aligner(config=config)

    # Need to remove `search_algorithm` and `aligner` since they are non-picklable pybind objects from librasr
    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["search_algorithm"] = None
        d["aligner"] = None

        return d

    # Recreate `search_algorithm` and `aligner` object from scratch since it's not picklable
    def __setstate__(self, d) -> None:
        self.__dict__ = d
        self.search_algorithm = self.init_search_algorithm()
        self.aligner = self.init_aligner()

    @abstractmethod
    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None: ...


class OfflineRasrRecogForwardStep(RasrRecogForwardStep):
    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        assert self.search_algorithm is not None

        raw_data = extern_data.as_raw_tensor_dict()
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        ctm_strs = []

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

            ms_per_enc_frame = round((seq_samples_size.item() / encoder_states.size(1)) / self.sample_rate * 1000)

            encoder_states = encoder_states.to(device="cpu")

            search_start = perf_counter()

            traceback = self.search_algorithm.recognize_segment(features=encoder_states)
            search_time = perf_counter() - search_start
            search_times.append(search_time)

            seq_time = _samples_to_seconds(seq_samples_size[0], self.sample_rate)

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)
            print(f'    Recognized: "{_traceback_to_transcription(traceback)}"')
            print("    Traceback:")
            for item in traceback:
                # if item.lemma.startswith("<") or item.lemma.startswith("["):
                #     continue
                print(f"        {repr(item)}")
            print(
                f"    Encoder time: {encoder_time:.3f} seconds, RTF {encoder_time / seq_time:.3f}, XRTF {seq_time / encoder_time:.3f}"
            )
            print(
                f"    Search time: {search_time:.3f} seconds, RTF {search_time / seq_time:.3f}, XRTF {seq_time / search_time:.3f}"
            )
            print()

            ctm_strs.append(_traceback_to_ctm_str(traceback, ms_per_enc_frame))

            if self.aligner is not None:
                align_traceback = self.aligner.align_segment(features=encoder_states, orth=orths[b] + " ")
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

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()

        ctm_string_tensor = Tensor(
            name="ctm_string",
            dtype="string",
            raw_tensor=np.array(ctm_strs, dtype="U"),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(ctm_string_tensor, name="ctm_string")

        if len(search_errors) > 0:
            search_errors_tensor = Tensor(
                name="search_errors",
                dtype="int32",
                raw_tensor=np.array(search_errors, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(search_errors_tensor, name="search_errors")

            model_errors_tensor = Tensor(
                name="model_errors",
                dtype="int32",
                raw_tensor=np.array(model_errors, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(model_errors_tensor, name="model_errors")

            skipped_tensor = Tensor(
                name="skipped",
                dtype="int32",
                raw_tensor=np.array(skipped, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(skipped_tensor, name="skipped")

            correct_tensor = Tensor(
                name="correct",
                dtype="int32",
                raw_tensor=np.array(correct, dtype=np.int32),
                feature_dim_axis=None,
                time_dim_axis=None,
            )
            run_ctx.mark_as_output(correct_tensor, name="correct")

        enc_time_tensor = Tensor(
            name="enc_time",
            dtype="float32",
            raw_tensor=np.array(encoder_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(enc_time_tensor, name="enc_time")

        audio_time_tensor = Tensor(
            name="audio_time",
            dtype="float32",
            raw_tensor=audio_samples_size / self.sample_rate,
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(audio_time_tensor, name="audio_time")

        search_time_tensor = Tensor(
            name="search_time",
            dtype="float32",
            raw_tensor=np.array(search_times, dtype=np.float32),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(search_time_tensor, name="search_time")


class StreamingRasrRecogForwardStep(RasrRecogForwardStep):
    def __init__(
        self,
        encoder_frame_shift_seconds: float,
        chunk_history_seconds: float,
        chunk_center_seconds: float,
        chunk_future_seconds: float,
        print_intermediate_outputs: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.chunk_history_samples = _seconds_to_samples(chunk_history_seconds, self.sample_rate)
        self.chunk_center_samples = _seconds_to_samples(chunk_center_seconds, self.sample_rate)
        self.chunk_future_samples = _seconds_to_samples(chunk_future_seconds, self.sample_rate)

        self.encoder_frame_shift_seconds = encoder_frame_shift_seconds

        self.print_intermediate_outputs = print_intermediate_outputs

    def __call__(self, *, model: EncoderModel, extern_data: TensorDict, **_) -> None:
        assert self.search_algorithm is not None

        raw_data = extern_data.as_raw_tensor_dict()
        audio_samples = raw_data["data"]
        audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
        orths = raw_data["raw"]
        seq_tags = raw_data["seq_tag"]

        ctm_strs = []

        unstable_latency_arrays = []
        unstable_latency_lengths = []
        stable_latency_arrays = []
        stable_latency_lengths = []

        for b in range(audio_samples.size(0)):
            self.search_algorithm.reset()
            self.search_algorithm.enter_segment()

            seq_samples_size = audio_samples_size[b].item()
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

                ms_per_enc_frame = round(
                    ((chunk_end_sample - chunk_start_sample) / encoder_states.size(1)) / self.sample_rate * 1000
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
                    stable_traceback = self.search_algorithm.get_common_prefix()

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

                    if self.print_intermediate_outputs:
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
            ctm_strs.append(_traceback_to_ctm_str(final_traceback, ms_per_enc_frame))

            print(f"Recognized sequence {repr(seq_tags[b])}")
            print(f'    Ground truth: "{orths[b]}"', flush=True)
            print(f'    Recognized: "{_traceback_to_transcription(final_traceback)}"')
            print("    Traceback:")
            for item in final_traceback:
                if item.lemma.startswith("<") or item.lemma.startswith("["):
                    continue
                print(f"        {repr(item)}")
            print()

            unstable_latency_arrays.append(np.array(unstable_latencies, dtype=np.float32))
            stable_latency_arrays.append(np.array(stable_latencies, dtype=np.float32))
            unstable_latency_lengths.append(len(unstable_latencies))
            stable_latency_lengths.append(len(stable_latencies))

        import returnn.frontend as rf

        run_ctx = rf.get_run_ctx()

        ctm_string_tensor = Tensor(
            name="ctm_string",
            dtype="string",
            raw_tensor=np.array(ctm_strs, dtype="U"),
            feature_dim_axis=None,
            time_dim_axis=None,
        )
        run_ctx.mark_as_output(ctm_string_tensor, name="ctm_string")

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
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs["unstable_latencies"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["unstable_latencies"].dims[
                1
            ].dyn_size_ext.raw_tensor = unstable_latency_lengths_array
        run_ctx.mark_as_output(unstable_latencies_tensor, name="unstable_latencies")

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
        if run_ctx.expected_outputs is not None:
            assert run_ctx.expected_outputs["stable_latencies"].dims[1].dyn_size_ext is not None
            run_ctx.expected_outputs["stable_latencies"].dims[1].dyn_size_ext.raw_tensor = stable_latency_lengths_array
        run_ctx.mark_as_output(stable_latencies_tensor, name="stable_latencies")


def recog_rasr_offline(
    descriptor: str,
    recog_rasr_config_file: tk.Path,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    encoder_serializers: Collection,
    sample_rate: int,
    params: OfflineRecogParameters,
    checkpoint: Optional[PtCheckpoint] = None,
    align_rasr_config_file: Optional[tk.Path] = None,
) -> RecogResult:
    compute_search_errors = align_rasr_config_file is not None
    model_outputs = {
        "ctm_string": {
            "dtype": "string",
            "feature_dim_axis": None,
            "time_dim_axis": None,
        },
        "audio_time": {
            "dtype": "float32",
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
    }
    forward_step_kwargs = [
        ("recog_rasr_config_file", DelayedFormat('tk.Path("{}")', recog_rasr_config_file)),
        ("sample_rate", sample_rate),
    ]

    if compute_search_errors:
        model_outputs.update(
            {
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
            }
        )
        forward_step_kwargs.append(("align_rasr_config_file", DelayedFormat('tk.Path("{}")', align_rasr_config_file)))

    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "raw": {"feature_dim_axis": None, "time_dim_axis": None, "dtype": "string"},
            },
            "model_outputs": model_outputs,
            "backend": "torch",
            "batch_size": 360 * sample_rate,
        },
        python_prolog=recipe_imports + [ExternalImport(rasr_binary_path)],
        python_epilog=[
            encoder_serializers,
            Import("sisyphus.tk"),
        ]
        + [
            Import(
                f"{OfflineSearchCallback.__module__}.{OfflineSearchCallback.__name__}",
                import_as="forward_callback",
            ),
            Import(
                f"{OfflineRasrRecogForwardStep.__module__}.{OfflineRasrRecogForwardStep.__name__}",
            ),
            Call(
                OfflineRasrRecogForwardStep.__name__,
                kwargs=forward_step_kwargs,
                return_assign_variables="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

    output_files = ["search_out.ctm", "rtf.py", "rasr.recog.log"]
    if compute_search_errors:
        output_files += ["search_errors.py", "rasr.align.log"]

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device="gpu" if params.gpu_mem_rqmt > 0 else "cpu",
        mem_rqmt=params.mem_rqmt,
        time_rqmt=168,
    )
    recog_job.add_alias(f"{recog_corpus.corpus_name}/{descriptor}")
    if params.gpu_mem_rqmt > 0:
        recog_job.rqmt["gpu_mem"] = params.gpu_mem_rqmt

    for output_file in output_files:
        tk.register_output(
            f"{recog_corpus.corpus_name}/{descriptor}/{output_file}",
            recog_job.out_files[output_file],
        )
    if compute_search_errors:
        search_error_job = ExtractSearchErrorDataJob(recog_job.out_files["search_errors.py"])
    rtf_job = ExtractRTFDataJob(recog_job.out_files["rtf.py"])

    score_job = recog_corpus.score_ctm(recog_job.out_files["search_out.ctm"])
    tk.register_output(f"{recog_corpus.corpus_name}/{descriptor}/scoring_reports", score_job.out_report_dir)

    extract_rasr_stats_job = ExtractRasrStatisticsJob(rasr_log_file=recog_job.out_files["rasr.recog.log"])

    tk.register_output(
        f"{recog_corpus.corpus_name}/{descriptor}/search_space_stats/out_step_hyps",
        extract_rasr_stats_job.out_step_hyps,
    )
    tk.register_output(
        f"{recog_corpus.corpus_name}/{descriptor}/search_space_stats/out_step_word_end_hyps",
        extract_rasr_stats_job.out_step_word_end_hyps,
    )
    tk.register_output(
        f"{recog_corpus.corpus_name}/{descriptor}/search_space_stats/out_step_trees",
        extract_rasr_stats_job.out_step_trees,
    )

    return RecogResult(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        wer=score_job.out_wer,
        deletion=score_job.out_percent_deletions,
        insertion=score_job.out_percent_insertions,
        substitution=score_job.out_percent_substitution,
        search_error_rate=search_error_job.out_search_error_rate if compute_search_errors else None,
        model_error_rate=search_error_job.out_model_error_rate if compute_search_errors else None,
        correct_rate=search_error_job.out_correct_rate if compute_search_errors else None,
        skipped_rate=search_error_job.out_skipped_rate if compute_search_errors else None,
        enc_rtf=rtf_job.out_enc_rtf,
        search_rtf=rtf_job.out_search_rtf,
        total_rtf=rtf_job.out_total_rtf,
        unstable_latency_stats=None,
        stable_latency_stats=None,
        step_hyps_stats=extract_rasr_stats_job.out_step_hyps,
        step_word_end_hyps_stats=extract_rasr_stats_job.out_step_word_end_hyps,
        step_trees_stats=extract_rasr_stats_job.out_step_trees,
    )


def recog_rasr_streaming(
    descriptor: str,
    recog_rasr_config_file: tk.Path,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    encoder_serializers: Collection,
    sample_rate: int,
    params: StreamingRecogParameters,
    checkpoint: Optional[PtCheckpoint] = None,
) -> RecogResult:
    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
                "raw": {"feature_dim_axis": None, "time_dim_axis": None, "dtype": "string"},
            },
            "model_outputs": {
                "ctm_string": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                    "time_dim_axis": None,
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
            "batch_size": 360 * sample_rate,
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
                    ("recog_rasr_config_file", DelayedFormat('tk.Path("{}")', recog_rasr_config_file)),
                    ("sample_rate", sample_rate),
                    ("encoder_frame_shift_seconds", params.encoder_frame_shift_seconds),
                    ("chunk_history_seconds", params.chunk_history_seconds),
                    ("chunk_center_seconds", params.chunk_center_seconds),
                    ("chunk_future_seconds", params.chunk_future_seconds),
                ],
                return_assign_variables="forward_step",
            ),
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

    output_files = ["search_out.ctm", "unstable_latencies.py", "stable_latencies.py", "rasr.recog.log"]

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device="gpu" if params.gpu_mem_rqmt > 0 else "cpu",
        mem_rqmt=params.mem_rqmt,
        time_rqmt=168,
    )
    recog_job.add_alias(f"{recog_corpus.corpus_name}/{descriptor}")
    if params.gpu_mem_rqmt > 0:
        recog_job.rqmt["gpu_mem"] = params.gpu_mem_rqmt

    for output_file in output_files:
        tk.register_output(
            f"{recog_corpus.corpus_name}/{descriptor}/{output_file}",
            recog_job.out_files[output_file],
        )

    unstable_latency_stats = ExtractStatisticsJob(recog_job.out_files["unstable_latencies.py"]).out_stats
    stable_latency_stats = ExtractStatisticsJob(recog_job.out_files["stable_latencies.py"]).out_stats

    tk.register_output(
        f"{recog_corpus.corpus_name}/{descriptor}/latencies/unstable_latency_stats", unstable_latency_stats
    )
    tk.register_output(f"{recog_corpus.corpus_name}/{descriptor}/latencies/stable_latency_stats", stable_latency_stats)

    score_job = recog_corpus.score_ctm(recog_job.out_files["search_out.ctm"])
    tk.register_output(f"{recog_corpus.corpus_name}/{descriptor}/scoring_reports", score_job.out_report_dir)

    extract_space_stats_job = ExtractRasrStatisticsJob(rasr_log_file=recog_job.out_files["rasr.recog.log"])
    tk.register_output(
        f"{recog_corpus.corpus_name}/{descriptor}/search_space_stats/out_step_hyps",
        extract_space_stats_job.out_step_hyps,
    )
    tk.register_output(
        f"{recog_corpus.corpus_name}/{descriptor}/search_space_stats/out_step_word_end_hyps",
        extract_space_stats_job.out_step_word_end_hyps,
    )
    tk.register_output(
        f"{recog_corpus.corpus_name}/{descriptor}/search_space_stats/out_step_trees",
        extract_space_stats_job.out_step_trees,
    )

    return RecogResult(
        descriptor=descriptor,
        corpus_name=recog_corpus.corpus_name,
        wer=score_job.out_wer,
        deletion=score_job.out_percent_deletions,
        insertion=score_job.out_percent_insertions,
        substitution=score_job.out_percent_substitution,
        search_error_rate=None,
        model_error_rate=None,
        correct_rate=None,
        skipped_rate=None,
        enc_rtf=None,
        search_rtf=None,
        total_rtf=None,
        unstable_latency_stats=unstable_latency_stats,
        stable_latency_stats=stable_latency_stats,
        step_hyps_stats=extract_space_stats_job.out_step_hyps,
        step_word_end_hyps_stats=extract_space_stats_job.out_step_word_end_hyps,
        step_trees_stats=extract_space_stats_job.out_step_trees,
    )
