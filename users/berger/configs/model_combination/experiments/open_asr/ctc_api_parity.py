"""Public-only replay of selected Granite/Speech-Qwen on the request-based API."""

import configparser
import hashlib
import json
from pathlib import Path

from sisyphus import tk
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common import ctc_llm, ctc_api_parity, rnnt_llm, qwen_speech, hf_benchmark, segment_resume
from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob


def request_api_config(text):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)
    cfg = RasrConfig()
    for section in parser.sections():
        for key, value in parser[section].items():
            parts = (section + "." + key).split(".")
            node = cfg
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
    search = cfg.lib_rasr.search_algorithm
    del search["word-start-marker"]
    search.word_piece_marker, search.word_piece_convention = "▁", "word-start-marked"
    search.log_stepwise_statistics = False
    return cfg


def run(datasets=("librispeech-dev-other",), *, transformers_version="5.16.1"):
    from ... import tools
    from . import scoring_version
    from ...model_pipelines.common.job_reports import register_job_report

    manifest = json.loads(Path(__file__).with_name("frozen_public_conditions.json").read_text())
    condition = manifest["conditions"]["Granite CTC | Speech Qwen 1.7B | one-pass"]
    # New job fingerprints use actual new runtime bytes, not legacy compatibility bytes.
    version = hashlib.sha256(
        b"".join(
            Path(m.__file__).read_bytes()
            for m in (ctc_llm, rnnt_llm, qwen_speech, hf_benchmark, ctc_api_parity, segment_resume)
        )
        + Path(__file__).read_bytes()
    ).hexdigest()
    result = {}
    for name in datasets:
        if name not in condition["development"]:
            raise ValueError("API parity is development-only")
        row = condition["development"][name]
        stage = row["stages"][0]
        p = stage["parameters"]
        baseline = row["public_artifacts"]
        required = [p[k] for k in ("dataset", "granite_repo", "speech_repo", "vocabulary", "prior")]
        required += [baseline[k] for k in ("references", "hypotheses", "score")]
        if not all(Path(v).exists() for v in required):
            raise FileNotFoundError("Public frozen parity artifacts unavailable; this is not an AppTek launcher")
        with ExperimentContext("api-parity/granite-speech-qwen-1.7b/" + name):
            binary = tools.get_llm_rnnt_search_rasr_binary_path()
            gate = ctc_llm.CtcLlmGateJob(binary, code_version=version)
            gate.add_alias("gate")
            cfg = request_api_config(stage["files"]["rasr_config"]["public_text"])
            config = WriteRasrConfigJob(cfg, RasrConfig()).out_config
            job = ctc_llm.CtcQwenRecogJob(
                dataset=tk.Path(p["dataset"]),
                repo=tk.Path(p["granite_repo"]),
                auxiliary_repo=tk.Path(p["speech_repo"]),
                vocabulary=tk.Path(p["vocabulary"]),
                binary=binary,
                config=config,
                gate=gate.out_report,
                kind="speech",
                code_version=version,
                prior=tk.Path(p["prior"]),
                prior_scale=p["prior_scale"],
                resume=True,
                transformers_version=transformers_version,
                speech_baseline=tk.Path(
                    manifest["conditions"]["Speech Qwen 1.7B | None | standalone"]["development"][name][
                        "public_artifacts"
                    ]["score"]
                ),
            )
            job.rqmt.update(gpu_mem=48, mem=64, time=24)
            job.add_alias("recognition")
            score = ScoreBenchmarkJob(
                tk.Path(baseline["references"]), job.out_hypotheses, code_version=scoring_version()
            )
            score.add_alias("scoring")
            comparison = ctc_api_parity.CompareCtcApisJob(
                tk.Path(baseline["hypotheses"]),
                job.out_hypotheses,
                tk.Path(baseline["score"]),
                score.out_summary,
                code_version=version,
            )
            comparison.add_alias("comparison")
            for label, path in [
                ("hypotheses", job.out_hypotheses),
                ("scoring", score.out_summary),
                ("comparison.json", comparison.out_report),
                ("differences", comparison.out_differences),
            ]:
                tk.register_output(label, path)
            register_job_report(
                "reports/asr-lm/ctc-api-parity-" + name + ".md",
                comparison.out_markdown,
                title="Granite CTC API parity: " + name,
            )
            result[name] = dict(recognition=job, scoring=score, comparison=comparison)
    return result
