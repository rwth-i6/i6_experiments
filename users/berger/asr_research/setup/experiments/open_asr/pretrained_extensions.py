"""Standalone beam controls and development-only pretrained correction grid."""

import hashlib
from pathlib import Path

from sisyphus import tk
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob

from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common import pretrained_correction as correction
from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
from . import scoring_version
from .runtime_budget import apply_runtime_budget


def _register(group, condition, name, prepared, factory):
    with ExperimentContext(f"open-asr/{group}/{condition}/{name}"):
        job = factory()
        apply_runtime_budget(job, name)
        job.add_alias("recognition")
        score = ScoreBenchmarkJob(prepared.out_references, job.out_hypotheses, code_version=scoring_version())
        score.add_alias("scoring")
        for key, value in (("scoring", score.out_summary), ("hypotheses", job.out_hypotheses),
                           ("runtime", job.out_runtime), ("segments", score.out_segments)):
            tk.register_output(key, value)
        if hasattr(job, "out_correction_validation"):
            tk.register_output("correction-validation", job.out_correction_validation)
        if hasattr(job, "out_log"):
            tk.register_output("rasr.log", job.out_log)
    if name in {"loquacious-dev", "loquacious-test"}:
        from ...data.loquacious_benchmark import FILTERS, LoquaciousSubsetJob
        from ...data.loquacious.datasets import _get_dev_short_segments
        for subset in list(FILTERS) + (["short"] if name.endswith("-dev") else []):
            filtered = LoquaciousSubsetJob(
                prepared.out_references, job.out_hypotheses, subset,
                segments=_get_dev_short_segments() if subset == "short" else None,
                manifest=prepared.out_manifest,
            )
            with ExperimentContext(f"open-asr/{group}/{condition}/{name}.{subset}"):
                sub_score = ScoreBenchmarkJob(filtered.out_references, filtered.out_hypotheses,
                                              code_version=scoring_version())
                sub_score.add_alias("scoring")
                tk.register_output("scoring", sub_score.out_summary)
    return job, score


def _version(base=""):
    return hashlib.sha256(base.encode() + Path(correction.__file__).read_bytes()).hexdigest()


def _kwargs(job, names, **overrides):
    result = {name: getattr(job, name) for name in names.split()}
    result.update(overrides)
    result["code_version"] = _version(job.code_version)
    return result


def run_standalone(data, *, speech_size="0.6b"):
    from .qwen_speech import SPEECH_MODELS
    from ...data.huggingface import DownloadHuggingFaceRepoJob
    repo_id, revision = SPEECH_MODELS[speech_size]
    speech = DownloadHuggingFaceRepoJob(repo_id, revision=revision).out_repo_dir
    result = []
    for name, prepared in data.items():
        # Standard native beam scoring on all corpora; an unnormalized-score
        # length-penalty ablation is development-only.
        penalties = (0.0, 1.0) if name in {"librispeech-dev-other", "loquacious-dev.short"} else (1.0,)
        for beam in (4, 8):
            for penalty in penalties:
                label = f"native-beam{beam}" + ("-lp0" if penalty == 0 else "")
                pair = _register(f"qwen-speech-{speech_size}", label, name, prepared, lambda: correction.SpeechQwenBeamJob(
                    dataset=prepared.out_dataset, speech_repo=speech, code_version=_version(),
                    beam=beam, length_penalty=penalty, batch_size=8,
                ))
                if speech_size == "1.7b":
                    pair[0].rqmt.update(gpu_mem=48, mem=96)
                result.append(pair)
    return result


def run_corrections(data=None):
    from . import qwen, qwen_speech, rnnt_qwen
    from .qwen_tuning import development_data
    if data is None:
        data = development_data()
    if set(data) - {"librispeech-dev-other", "loquacious-dev.short"}:
        raise ValueError("Prior/ILM correction tuning is development-only")
    granite = qwen.run(data=data, two_pass=False, matched_controls=True)
    speech = qwen_speech.run(data=data, tuning=True)
    rnnt = rnnt_qwen.run(data=data, speech_baselines=speech["scores"]["qwen-speech/native"],
                        tuning_only=True, focused_tuning=True)
    result = []
    for name, prepared in data.items():
        def base(suite, key):
            return suite["scores"][key][name].creator.hypotheses.creator
        ctc = base(granite, "granite-ctc/one-pass")
        ctc_speech = base(speech, "qwen-speech/granite-fusion")
        ctc_zero = base(granite, "granite-ctc/zero-beam32-thr6")
        rtext = base(rnnt, "parakeet-rnnt/text-qwen-beam8-thr2")
        rspeech = base(rnnt, "parakeet-rnnt/speech-qwen-scale-0.1-beam8-pre32-thr3")
        rzero = base(rnnt, "parakeet-rnnt/zero-beam32")
        source = dict(repo="openslr/librispeech_asr", config="clean", split="train.100", audio_key="audio",
                      revision="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1") if name.startswith("librispeech") else dict(
                      repo="speechbrain/LoquaciousSet", config="small", split="train", audio_key="wav",
                      revision="0e84cdb9e4b826afaabca5d33ec9453b11aacef3")
        with ExperimentContext(f"open-asr/pretrained-correction/{name}"):
            prior = correction.EstimateGranitePriorJob(ctc.repo, source, code_version=_version())
            prior.add_alias("estimate-prior")
            tk.register_output("prior", prior.out_prior)
        for weight in (0.1, 0.2):
            cfg = qwen.dynamic_config(ctc.vocabulary, ctc.qwen_repo, scale=0.0)
            cfg.lib_rasr.search_algorithm.llm.type = "qwen-dynamic-zero"
            cfg.lib_rasr.search_algorithm.word_penalty = 0.0
            config = WriteRasrConfigJob(cfg, RasrConfig()).out_config
            kw = _kwargs(ctc_zero, "dataset repo qwen_repo vocabulary binary rasr_config gate code_version zero",
                         rasr_config=config)
            result.append(_register("qwen-granite", f"zero-beam32-thr6-prior{weight:g}", name, prepared,
                                    lambda: correction.CorrectedGraniteQwenJob(prior=prior.out_prior, prior_scale=weight, **kw)))
            raw = rzero.vocabulary.creator
            cfg = rnnt_qwen.recognition_config(raw, kind="zero", beam=8, pre_beam=32, threshold=2.0)
            kw = _kwargs(rzero, "dataset repo auxiliary_repo vocabulary binary config gate kind code_version speech_baseline",
                         config=WriteRasrConfigJob(cfg, RasrConfig()).out_config)
            result.append(_register("qwen-parakeet-rnnt", f"zero-beam8-thr2-ilm{weight:g}", name, prepared,
                                    lambda: correction.CorrectedRnntQwenJob(ilm_scale=weight, **kw)))
            for scale in (0.1, 0.3, 0.5):
                cfg = qwen.dynamic_config(ctc.vocabulary, ctc.qwen_repo, scale=scale)
                kw = _kwargs(ctc, "dataset repo qwen_repo vocabulary binary rasr_config gate code_version",
                             rasr_config=WriteRasrConfigJob(cfg, RasrConfig()).out_config)
                label = f"one-pass-scale-{scale:g}-prior{weight:g}"
                result.append(_register("qwen-granite", label, name, prepared,
                                        lambda: correction.CorrectedGraniteQwenJob(prior=prior.out_prior, prior_scale=weight, **kw)))
                cfg = qwen_speech.fusion_config(ctc_speech.vocabulary, scale=scale)
                kw = _kwargs(ctc_speech, "dataset speech_repo code_version granite_repo binary vocabulary rasr_config baseline max_new_tokens batch_size",
                             rasr_config=WriteRasrConfigJob(cfg, RasrConfig()).out_config)
                label = f"granite-fusion-scale-{scale:g}-prior{weight:g}"
                result.append(_register("qwen-speech-0.6b", label, name, prepared,
                                        lambda: correction.CorrectedGraniteSpeechJob(prior=prior.out_prior, prior_scale=weight, **kw)))
                for kind, template, threshold in (("text", rtext, 2.0), ("speech", rspeech, 3.0)):
                    cfg = rnnt_qwen.recognition_config(raw, kind=kind, auxiliary_repo=template.auxiliary_repo,
                                                       scale=scale, beam=8, pre_beam=32, threshold=threshold)
                    kw = _kwargs(template, "dataset repo auxiliary_repo vocabulary binary config gate kind code_version speech_baseline",
                                 config=WriteRasrConfigJob(cfg, RasrConfig()).out_config)
                    label = f"{kind}-qwen-scale-{scale:g}-beam8-pre32-thr{threshold:g}-ilm{weight:g}"
                    result.append(_register("qwen-parakeet-rnnt", label, name, prepared,
                                            lambda: correction.CorrectedRnntQwenJob(ilm_scale=weight, **kw)))
    return result


def run():
    from .qwen_tuning import development_data
    data = development_data()
    return dict(standalone=run_standalone(data), corrections=run_corrections(data))
