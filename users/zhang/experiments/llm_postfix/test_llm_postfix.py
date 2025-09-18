from dataclasses import dataclass
import gzip
import json
import logging
import os
from typing import Iterator, Optional, cast
from apptek_asr.artefacts import AbstractArtefactRepository
from apptek_asr.artefacts.factory import ArtefactSpecification
from apptek_asr.artefacts.runtime.repo import SingularityRuntimeV2
from apptek_asr.lib.parsers import parse_segmented_ctm, AppTekSegment
from apptek_asr.meta.evaluator import AsrMonEvaluatorV1
from apptek_asr.software.venv import CreatePythonVEnvV2Job
from apptek_llm.inference.openai.provider import OpenAIAPIProvider, VLLMAPIProvider
from apptek_llm.inference.openai.api import OpenAIChatCompletionsAPIJob
from sisyphus import Job, gs, tk
from sisyphus.task import Task
from apptek_asr.report import TabularReport


OPENAI_MODEL_NAME = "gpt-4o"
VLLM_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
ENGINE = "openai"

rasr_name = "streaming-rasr-2025-03-18"
asrmon_name = "asrmon-2025-02-10"
sctk_name = "sctk-2022-09-08"
runtime_name = "ApptekCluster-ubuntu2204-tf2.17.1-pt2.6.0-2025-02-26"


class CtmToPromptJob(Job):
    def __init__(self, ctm_file: tk.Path, prompt_template: str, full_recording: bool = False) -> None:
        self.ctm_file = ctm_file
        self.prompt_template = prompt_template
        self.full_recording = full_recording

        self.out_prompts = self.output_path("prompts.txt.gz")

        self.rqmt = None

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt=self.rqmt, mini_task=self.rqmt is None)

    def run(self):
        ctm = parse_segmented_ctm(self.ctm_file.get_path())
        with gzip.open(self.out_prompts, "wt") as out_prompts:
            for recording_segments in ctm.values():
                if self.full_recording:
                    text = "\n".join(seg.orth for seg in recording_segments)
                    messages = [{"role": "user", "content": self.prompt_template.format(text)}]
                    json.dump({"messages": messages}, out_prompts)
                    out_prompts.write("\n")
                else:
                    for segment in recording_segments:
                        text = segment.orth
                        messages = [{"role": "user", "content": self.prompt_template.format(text)}]
                        json.dump({"messages": messages}, out_prompts)
                        out_prompts.write("\n")


class ChatCompletionToCtmJob(Job):
    def __init__(self, completions_file: tk.Path, reference_ctm: tk.Path, full_recording: bool = False) -> None:
        self.completions_file = completions_file
        self.reference_ctm = reference_ctm
        self.full_recording = full_recording

        self.out_ctm = self.output_path("processed.ctm")

        self.rqmt = None

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt=self.rqmt, mini_task=self.rqmt is None)

    def run(self):
        ctm = parse_segmented_ctm(self.reference_ctm.get_path())
        with open(self.completions_file, "rt") as in_completions_f, open(self.out_ctm, "wt") as out_ctm_f:
            out_ctm_f.write(";; <name> <track> <start> <duration> <word> <confidence>\n")
            for recording_name, recording_segments in ctm.items():
                if self.full_recording:
                    orth = " ".join(seg.orth for seg in recording_segments)
                    start = recording_segments[0].start
                    end = recording_segments[-1].end
                    channel = recording_segments[0].channel
                    if not all(seg.channel == channel for seg in recording_segments):
                        raise NotImplementedError("Full Recording handling with mixed channel not implemented")
                    recording_segments = [AppTekSegment(start, end, orth, channel)]
                for segment in recording_segments:
                    llm_update = json.loads(in_completions_f.readline())
                    try:
                        new_orth = llm_update["hypothesis"][0]["messages"][0]["content"]
                        if self.llm_update_rejection_heuristic(segment.orth, new_orth):
                            segment.orth = new_orth
                    except (KeyError, IndexError):
                        pass
                    out_ctm_f.write(self.segment_to_ctm(recording_name, segment))

    @staticmethod
    def segment_to_ctm(recording_name: str, segment: AppTekSegment) -> str:
        res = f";; {recording_name} {segment.start} {segment.end}\n"
        word_duration = (segment.end - segment.start) / len(words := segment.orth.split())
        for i, word in enumerate(words):
            res += f"{recording_name} {segment.channel} {segment.start + i * word_duration:.3f} {word_duration:.3f} {word} 1.0000\n"
        return res

    @staticmethod
    def llm_update_rejection_heuristic(orig_orth: str, new_orth: str) -> bool:
        if abs(len(orig_orth) - len(new_orth)) / len(orig_orth) > 0.25:
            return False
        return True


@dataclass
class SamplingArgs:
    max_completion_tokens: Optional[int] = None
    n: int = 1
    seed: Optional[int] = None
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0


def py():
    aar = AbstractArtefactRepository()
    runtime = cast(
        SingularityRuntimeV2,
        aar.get_artefact_factory("runtime", runtime_name).build(),
    )
    gs.worker_wrapper = runtime.worker_wrapper

    # prepare env and keys
    openai_key = "UNSET"
    if os.path.exists(keyfile := os.path.expanduser("~/.config/openai_key")):
        with open(keyfile, "rt") as keyfile_f:
            openai_key = keyfile_f.readline().strip()
    else:
        logging.warning("API key unset")

    dummy_python_env = CreatePythonVEnvV2Job(packages=[["pip"]], venv_extra_args=["--system-site-package"])
    vllm_python_env = CreatePythonVEnvV2Job(packages=[["vllm", "bitsandbytes>=0.45.3"]])
    vllm_provider = VLLMAPIProvider(
        python_bin=vllm_python_env.out_python_bin,
        model_name_or_path=VLLM_MODEL_NAME,
        vllm_extra_args=["--device", "cuda", "--quantization", "bitsandbytes", "--load_format", "bitsandbytes"],
    )
    openai_provider = OpenAIAPIProvider(OPENAI_MODEL_NAME)

    # get ctm file
    evaluator = AsrMonEvaluatorV1(
        aar=aar,
        rasr_spec=ArtefactSpecification("rasr", rasr_name),
        prod_model_spec=ArtefactSpecification(
            "prod.asrmon_v1.EN.mbw",
            "2024-10-asrmon-quantized-onnx-streaming-finetuned-1B-conformer-epoch_22-quantized-onnx-lstmlm-v1",
        ),
        runtime_spec=ArtefactSpecification("runtime", runtime_name),
        sctk_spec=ArtefactSpecification("software.sctk", sctk_name),
        asrmon_spec=ArtefactSpecification("software.asrmon", asrmon_name),
    )
    eval_set_specs = {
        eval_set_name: ArtefactSpecification("test_set.EN_US.f16kHz", eval_set_name)
        for eval_set_name in [
            "dev_news_202203-v3",
            "dev_keynote_202207-v3",
            "dev_meetings_202207-v3",
            "dev_movies_tvshows_202207-v3",
        ]
    }
    eval_set_specs["dev_chinese_general_telephony_heldout-v1"] = ArtefactSpecification(
        "test_set.EN_US.f8kHz", "dev_chinese_general_telephony_heldout-v1"
    )

    report = TabularReport()
    report.add_row("dev set", "prompt", "FFWER")
    SEPARATOR_ROW = ["-" * len(c) for c in ["dev_chinese_general_telephony_heldout-v1", "names_focus", "FFWER"]]
    report.separator = " | "

    for eval_set_name, eval_set_spec in eval_set_specs.items():
        report.add_row(*SEPARATOR_ROW)
        # original recognition + scoring
        recog_name, ms, orig_reports = evaluator.run(eval_set_spec=eval_set_spec, get_reports=True)
        report.add_row(eval_set_name, "-", ms["full_file_wer"])
        for key, value in orig_reports.items():
            tk.register_output(f"recog_{eval_set_name}_original_{key}_reports", value)

        # prepare prompts
        ctm_path = evaluator.jobs[f"{recog_name}_recog"].out_ctm_file
        ctm_prompts = {
            "corrected": CtmToPromptJob(
                ctm_file=ctm_path,
                prompt_template=(
                    "You are second pass transcriber that is given the first pass transcription of an acoustic utterance. "
                    "Please fix any transcription errors of the first pass transcriber to minimize the edit distance to the "
                    "unknown reference transcription. Write only the updated sentence without any additional comments. "
                    "Write only lowercased words without punctuation\n\nTranscript:\n{}"
                ),
                full_recording=False,
            ),
            "names_focus": CtmToPromptJob(
                ctm_file=ctm_path,
                prompt_template=(
                    "You are second pass transcriber that is given the first pass transcription of an acoustic utterance. "
                    "Please fix any transcription errors of the first pass transcriber to minimize the edit distance to the "
                    "unknown reference transcription. Note that the original was spoken, so please do not correct disfluencies "
                    "in the text and rather focus on correcting proper names. Write only the updated sentence without any additional comments. "
                    "Write only lowercased words without punctuation\n\nTranscript:\n{}"
                ),
                full_recording=False,
            ),
        }

        for name, prompt_job in ctm_prompts.items():
            # run LLM
            ctm_completions = OpenAIChatCompletionsAPIJob(
                data_path=prompt_job.out_prompts,
                provider=vllm_provider if ENGINE == "vllm" else openai_provider,
                sampling_args=SamplingArgs(),
                venv_path=dummy_python_env.out_venv,
            )
            if ENGINE == "openai":
                ctm_completions.set_env("OPENAI_API_KEY", openai_key)
            else:
                ctm_completions.set_env(
                    "PATH",
                    "/opt/apptek/thirdparty/usr/bin:/opt/apptek/thirdparty/usr/bin:"
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                )
                ctm_completions.set_env("LD_LIBRARY_PATH", "/.singularity.d/libs")
                ctm_completions.rqmt["sbatch_args"] = ["-p", "gpu-48g"]  # type: ignore

            # convert to ctm
            new_ctm = ChatCompletionToCtmJob(ctm_completions.out, reference_ctm=ctm_path, full_recording=False)

            # scoring
            with runtime.environment():
                eval_set = eval_set_spec.build(aar)
                sctk = aar.get_artefact_factory("software.sctk", sctk_name).build()
                ms, updated_reports = eval_set["metrics"].run(aar, new_ctm.out_ctm, extra_scorer_kwargs=sctk)

            report.add_row(eval_set_name, name, ms["full_file_wer"])
            for key, value in updated_reports.items():
                tk.register_output(f"recog_{eval_set_name}_{name}_{key}_reports", value)

    tk.register_report("report", report)
