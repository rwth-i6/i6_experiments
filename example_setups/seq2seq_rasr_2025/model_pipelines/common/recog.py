__all__ = ["recog_base", "recog_flashlight", "recog_rasr"]

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Iterator, List, Literal, Optional, Protocol

import numpy as np
import torch
from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.returnn.search import SearchBPEtoWordsJob, SearchWordsToCTMJob
from i6_experiments.common.setups.serialization import Collection, ExternalImport, Import, PartialImport
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import Tensor, TensorDict
from sisyphus import Job, Task, tk

from ...data.base import DataConfig
from ...tools import returnn_python_exe, returnn_root
from .corpus import ScorableCorpus
from .imports import recipe_imports

# -------------------
# --- Base ----------
# -------------------


@dataclass
class RecogResult:
    descriptor: str
    corpus_name: str
    wer: tk.Variable
    deletion: tk.Variable
    insertion: tk.Variable
    substitution: tk.Variable
    enc_rtf: tk.Variable
    search_rtf: tk.Variable
    total_rtf: tk.Variable


class SearchCallback(ForwardCallbackIface):
    def init(self, *, model):
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


class EncoderModel(Protocol):
    def forward(self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor) -> torch.Tensor: ...


def _base_recog_forward_step(
    *,
    model: EncoderModel,
    extern_data: TensorDict,
    search_function: SearchFunction,
    sample_rate: int = 16000,
):
    raw_data = extern_data.as_raw_tensor_dict()
    audio_samples = raw_data["data"]
    audio_samples_size = raw_data["data:size1"].to(device=audio_samples.device)
    seq_tags = raw_data["seq_tag"]

    tokens_arrays = []
    token_lengths = []

    encoder_times = []
    search_times = []

    # Since the first run takes longer, run it before with dummy data to not distort RTF numbers
    search_function(
        features=model.forward(
            torch.zeros(1, sample_rate, 1, device=audio_samples.device),
            torch.full([1], fill_value=sample_rate, device=audio_samples.device),
        ).to(device="cpu")
    )

    for b in range(audio_samples.size(0)):
        seq_samples_size = audio_samples_size[b : b + 1]
        seq_samples = audio_samples[b : b + 1, : seq_samples_size[0]]  # [1, T, 1]

        encoder_start = perf_counter()
        encoder_states = model.forward(seq_samples, seq_samples_size)
        encoder_time = perf_counter() - encoder_start
        encoder_times.append(encoder_time)

        encoder_states = encoder_states.to(device="cpu")

        search_start = perf_counter()
        result: str = search_function(features=encoder_states)
        search_time = perf_counter() - search_start
        search_times.append(search_time)

        result = result.replace("</s>", "")
        result = result.replace("<blank>", "")

        tokens_array = np.array(result.split(), dtype="U")
        tokens_arrays.append(tokens_array)
        token_lengths.append(len(tokens_array))

        seq_time = seq_samples_size[0] / sample_rate

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


def recog_base(
    descriptor: str,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    model_serializers: Collection,
    forward_step_import: Import,
    device: Literal["cpu", "gpu"] = "cpu",
    checkpoint: Optional[PtCheckpoint] = None,
    extra_output_files: Optional[List[str]] = None,
) -> RecogResult:
    recog_returnn_config = ReturnnConfig(
        config={
            "extern_data": {
                "data": {"dim": 1, "dtype": "float32"},
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
        python_prolog=recipe_imports
        + [
            ExternalImport(
                tk.Path(
                    "/work/asr4/berger/rasr_dev/label_scorer/rasr/lib/linux-x86_64-standard",
                    hash_overwrite="RASR_BINARY_PATH",
                )
            )
        ],
        python_epilog=[
            model_serializers,
            Import(
                f"{SearchCallback.__module__}.{SearchCallback.__name__}",
                import_as="forward_callback",
            ),
            forward_step_import,
        ],  # type: ignore
        sort_config=False,
    )

    recog_returnn_config.update(recog_data_config.get_returnn_data("forward_data"))

    output_files = ["search_out.py", "rtf.py"] + (extra_output_files or [])

    recog_job = ReturnnForwardJobV2(
        model_checkpoint=checkpoint,
        returnn_config=recog_returnn_config,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_files=output_files,
        device=device,
        mem_rqmt=16,
        time_rqmt=168,
    )
    recog_job.add_alias(f"recognition/{recog_corpus.corpus_name}/{descriptor}")

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

    return RecogResult(
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


# -------------------
# --- Flashlight ----
# -------------------


@lru_cache(maxsize=1)
def _get_flashlight_search_function(
    *,
    vocab_file: tk.Path,
    lm_weight: float = 0.0,
    blank_token: str = "<blank>",
    silence_token: str = "<blank>",
    unk_word: str = "<unk>",
    **kwargs,
) -> SearchFunction:
    from torchaudio.models.decoder import ctc_decoder

    vocab = Vocabulary.create_vocab(vocab_file=vocab_file, unknown_label=None)
    assert vocab._vocab is not None
    labels = list({value: key for key, value in vocab._vocab.items()}.values())

    if "<blank>" not in labels:
        labels.append("<blank>")

    print(f"labels: {labels}")

    decoder = ctc_decoder(
        tokens=labels,
        lm_weight=lm_weight,
        blank_token=blank_token,
        silence_token=silence_token,
        unk_word=unk_word,
        **kwargs,
    )

    def wrapper(features: torch.Tensor) -> str:
        nonlocal labels
        nonlocal decoder

        hyps = decoder(-features)
        str_result = " ".join([labels[token] for token in hyps[0][0].tokens])
        return str_result

    return wrapper


def _flashlight_recog_forward_step(
    *,
    model: EncoderModel,
    extern_data: TensorDict,
    sample_rate: int = 16000,
    **kwargs,
):
    search_function = _get_flashlight_search_function(**kwargs)
    return _base_recog_forward_step(
        model=model, extern_data=extern_data, search_function=search_function, sample_rate=sample_rate
    )


def recog_flashlight(
    descriptor: str,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    model_serializers: Collection,
    sample_rate: int = 16000,
    device: Literal["cpu", "gpu"] = "cpu",
    checkpoint: Optional[PtCheckpoint] = None,
    **kwargs,
) -> RecogResult:
    flashlight_forward_step_import = PartialImport(
        code_object_path=f"{_flashlight_recog_forward_step.__module__}.{_flashlight_recog_forward_step.__name__}",
        unhashed_package_root="",
        hashed_arguments={**kwargs, "sample_rate": sample_rate},
        unhashed_arguments={},
        import_as="forward_step",
    )
    return recog_base(
        descriptor=descriptor,
        recog_data_config=recog_data_config,
        recog_corpus=recog_corpus,
        model_serializers=model_serializers,
        forward_step_import=flashlight_forward_step_import,
        device=device,
        checkpoint=checkpoint,
    )


# -------------------
# --- RASR ----------
# -------------------


@lru_cache(maxsize=1)
def _get_rasr_search_function(config_file: tk.Path) -> SearchFunction:
    from librasr import Configuration, SearchAlgorithm

    config = Configuration()
    config.set_from_file(config_file)

    search_algorithm = SearchAlgorithm(config=config)
    return search_algorithm.recognize_segment


def _rasr_recog_forward_step(
    *,
    model: EncoderModel,
    extern_data: TensorDict,
    config_file: tk.Path,
    sample_rate: int = 16000,
    **_,
) -> None:
    _base_recog_forward_step(
        model=model,
        extern_data=extern_data,
        search_function=_get_rasr_search_function(config_file),
        sample_rate=sample_rate,
    )


def recog_rasr(
    descriptor: str,
    recog_data_config: DataConfig,
    recog_corpus: ScorableCorpus,
    model_serializers: Collection,
    rasr_config_file: tk.Path,
    sample_rate: int = 16000,
    device: Literal["cpu", "gpu"] = "cpu",
    checkpoint: Optional[PtCheckpoint] = None,
) -> RecogResult:
    rasr_forward_step_import = PartialImport(
        code_object_path=f"{_rasr_recog_forward_step.__module__}.{_rasr_recog_forward_step.__name__}",
        unhashed_package_root="",
        hashed_arguments={
            "config_file": rasr_config_file,
            "sample_rate": sample_rate,
        },
        unhashed_arguments={},
        import_as="forward_step",
    )
    return recog_base(
        descriptor=descriptor,
        recog_data_config=recog_data_config,
        recog_corpus=recog_corpus,
        model_serializers=model_serializers,
        forward_step_import=rasr_forward_step_import,
        device=device,
        checkpoint=checkpoint,
        extra_output_files=["rasr.log"],
    )
