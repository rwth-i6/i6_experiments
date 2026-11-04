"""Development-only Speech-Qwen balance, prior, rescoring and pruning batch."""

import hashlib
from pathlib import Path

from sisyphus import tk
from i6_core.rasr.config import RasrConfig, WriteRasrConfigJob

from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common import speech_qwen_round as runtime
from ...model_pipelines.common import pretrained_correction as correction
from ...model_pipelines.common.benchmark_scoring import ScoreBenchmarkJob
from . import scoring_version
from .pretrained_extensions import _register, _version


def development(data=None):
    if data is None:
        from .qwen_tuning import development_data
        data = development_data()
    if not data or set(data) - {"librispeech-dev-other", "loquacious-dev.short"}:
        raise ValueError("Speech-Qwen round accepts dev-other/dev.short only")
    return data


def bases(data):
    from . import qwen_speech, rnnt_qwen
    speech = qwen_speech.run(data=data, tuning=True, speech_size="1.7b")
    rnnt = rnnt_qwen.run(data=data, speech_baselines=speech["scores"]["qwen-speech/native"],
                         speech_size="1.7b", speech_only=True, tuning_only=True)
    return speech, rnnt


def template(result, key, name):
    return result["scores"][key][name].creator.hypotheses.creator


def rnnt_kwargs(job, config):
    keys = "dataset repo auxiliary_repo vocabulary binary gate kind code_version speech_baseline"
    return dict({k: getattr(job, k) for k in keys.split()}, config=config)


def speech_kwargs(job, config):
    keys = "dataset speech_repo code_version granite_repo binary vocabulary baseline max_new_tokens batch_size"
    return dict({k: getattr(job, k) for k in keys.split()}, rasr_config=config)


def gpu48(pair):
    pair[0].rqmt.update(gpu_mem=48, mem=96)
    return pair


def run_fusion(data=None):
    from . import qwen_speech, rnnt_qwen
    data = development(data)
    speech, rnnt = bases(data)
    result = []
    for name, prepared in data.items():
        r = template(rnnt, "parakeet-rnnt/speech-qwen-fast", name)
        for scale in (.05, .1, .15, .2, .25, .3):
            for penalty in (0.0, -.5, -1.0):
                cfg = rnnt_qwen.recognition_config(r.vocabulary.creator, kind="speech", auxiliary_repo=r.auxiliary_repo,
                                                   scale=scale, beam=8, pre_beam=32, threshold=3)
                cfg.lib_rasr.search_algorithm.word_penalty = penalty
                kw = rnnt_kwargs(r, WriteRasrConfigJob(cfg, RasrConfig()).out_config)
                condition = f"speech-qwen-scale-{scale:g}-word-penalty{penalty:g}-beam8-pre32-thr3"
                result.append(gpu48(_register("qwen-parakeet-rnnt-speech-1.7b", condition, name, prepared,
                                             lambda: runtime.RnntQwenRecogJob(**kw))))
        c = template(speech, "qwen-speech/granite-fusion", name)
        source = dict(repo="openslr/librispeech_asr", config="clean", split="train.100", audio_key="audio",
                      revision="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1") if name.startswith("librispeech") else dict(
                      repo="speechbrain/LoquaciousSet", config="small", split="train", audio_key="wav",
                      revision="0e84cdb9e4b826afaabca5d33ec9453b11aacef3")
        with ExperimentContext(f"open-asr/speech-qwen-round/prior/{name}"):
            prior = correction.EstimateGranitePriorJob(c.granite_repo, source, code_version=_version())
            prior.add_alias("estimation")
            tk.register_output("prior", prior.out_prior)
        for weight in (0.0, .05, .1, .2):
            for scale in (.2, .3, .4):
                cfg = qwen_speech.fusion_config(c.vocabulary, scale=scale)
                kw = speech_kwargs(c, WriteRasrConfigJob(cfg, RasrConfig()).out_config)
                condition = f"granite-fusion-scale-{scale:g}-prior{weight:g}"
                def make():
                    if not weight:
                        from ...model_pipelines.common.qwen_speech import QwenSpeechRecogJob
                        return QwenSpeechRecogJob(**kw)
                    kw["code_version"] = _version(c.code_version)
                    return correction.CorrectedGraniteSpeechJob(prior=prior.out_prior, prior_scale=weight, **kw)
                result.append(gpu48(_register("qwen-speech-1.7b", condition, name, prepared, make)))
    return result


def run_pruning(data=None, *, stage="threshold", threshold=None, scale=None, word_penalty=0.0):
    """Thresholds first at generous beams; beam tightening is a later opt-in.

    No automatic test-set selection. Beam stage requires explicit dev-selected
    scale/threshold (and optionally word penalty) after the first stage finishes.
    """
    from . import rnnt_qwen
    data = development(data)
    if stage not in {"threshold", "beam"} or (stage == "beam" and (threshold is None or scale is None)):
        raise ValueError("Beam stage requires selected development scale and threshold")
    _, rnnt = bases(data)
    result = []
    settings = [(s, t, 32, 128) for s in (.1, .3) for t in (1.5, 2, 2.5, 3)] if stage == "threshold" else [
        (scale, threshold, b, p) for b, p in ((4, 16), (8, 32), (16, 64), (32, 128))]
    for name, prepared in data.items():
        r = template(rnnt, "parakeet-rnnt/speech-qwen-fast", name)
        for lm_scale, thr, beam, pre in settings:
            cfg = rnnt_qwen.recognition_config(r.vocabulary.creator, kind="speech", auxiliary_repo=r.auxiliary_repo,
                                               scale=lm_scale, beam=beam, pre_beam=pre, threshold=thr)
            cfg.lib_rasr.search_algorithm.word_penalty = word_penalty
            kw = rnnt_kwargs(r, WriteRasrConfigJob(cfg, RasrConfig()).out_config)
            condition = (f"speech-qwen-prune-{stage}-scale-{lm_scale:g}-word-penalty{word_penalty:g}"
                         f"-beam{beam}-pre{pre}-thr{thr:g}")
            result.append(gpu48(_register("qwen-parakeet-rnnt-speech-1.7b", condition, name, prepared,
                                         lambda: runtime.RnntQwenRecogJob(**kw))))
    return result


def export_score(group, condition, name, prepared, hypotheses, runtime_path=None):
    with ExperimentContext(f"open-asr/{group}/{condition}/{name}"):
        score = ScoreBenchmarkJob(prepared.out_references, hypotheses, code_version=scoring_version())
        score.add_alias("scoring")
        tk.register_output("scoring", score.out_summary)
        tk.register_output("segments", score.out_segments)
        tk.register_output("hypotheses", hypotheses)
        if runtime_path is not None:
            tk.register_output("runtime", runtime_path)
    return score


def selection_grid():
    return [(50, s, p, policy) for s in (.1, .3, .5) for p in (0.0, -1.0)
            for policy in ("original", "initial-capital-period")] + [(10, .3, 0.0, "original")]


def run_rescoring(data=None):
    from . import qwen, qwen_speech, rnnt_qwen
    from ...data.huggingface import DownloadHuggingFaceRepoJob
    from ...model_pipelines.common import qwen_speech as speech_runtime, rnnt_llm, hf_benchmark
    data = development(data)
    speech, rnnt = bases(data)
    granite = qwen.run(data=data, two_pass=True)
    version = hashlib.sha256(b"".join(Path(m.__file__).read_bytes()
                                     for m in (runtime, speech_runtime, rnnt_llm, hf_benchmark))).hexdigest()
    generation, scoring, selections, oracles = [], [], [], []
    for name, prepared in data.items():
        r = template(rnnt, "parakeet-rnnt/speech-qwen-fast", name)
        cfg = rnnt_qwen.recognition_config(r.vocabulary.creator, kind="zero", beam=64, pre_beam=128, threshold=6)
        # N-best compares concrete path costs, not marginal sums of alignments.
        cfg.lib_rasr.search_algorithm.recombination_mode = "viterbi"
        with ExperimentContext(f"open-asr/speech-qwen-round/parakeet-candidates/{name}"):
            gate = runtime.RnntSpeechNbestGateJob(r.binary, code_version=version)
            gate.add_alias("native-nbest-gate")
            tk.register_output("gate", gate.out_report)
            kw = rnnt_kwargs(r, WriteRasrConfigJob(cfg, RasrConfig()).out_config)
            kw.update(kind="zero", auxiliary_repo=None, speech_baseline=None, gate=gate.out_report,
                      code_version=hashlib.sha256((version + r.code_version).encode()).hexdigest())
            rnnt_gen = runtime.RnntSpeechNbestJob(n=50, **kw)
            rnnt_gen.rqmt.update(gpu_mem=48, mem=96)
            rnnt_gen.add_alias("generation")
            tk.register_output("candidates", rnnt_gen.out_candidates)
            tk.register_output("validation", rnnt_gen.out_validation)
            tk.register_output("rasr.log", rnnt_gen.out_log)
        selected = template(granite, "granite-ctc/two-pass-librispeech-lexicon", name)
        ctc_gen = selected.candidates.creator.candidates.creator
        generation.extend([ctc_gen, rnnt_gen])
        for family, gen, first, gen_runtime in (
            ("granite", ctc_gen, ctc_gen.out_first_pass, ctc_gen.out_report),
            ("parakeet", rnnt_gen, rnnt_gen.out_hypotheses, rnnt_gen.out_runtime),
        ):
            for size in ("0.6b", "1.7b"):
                group = f"qwen-speech-{size}" if family == "granite" else (
                    "qwen-parakeet-rnnt" if size == "0.6b" else "qwen-parakeet-rnnt-speech-1.7b")
                repo, revision = qwen_speech.SPEECH_MODELS[size]
                speech_repo = DownloadHuggingFaceRepoJob(repo, revision=revision).out_repo_dir
                export_score(group, "first-pass-speech-nbest", name, prepared, first, gen_runtime)
                with ExperimentContext(f"open-asr/{group}/speech-nbest-scores/{name}"):
                    lm = runtime.ScoreSpeechNbestJob(prepared.out_dataset, gen.out_candidates, speech_repo,
                                                     code_version=version)
                    lm.add_alias("scoring")
                    tk.register_output("scores", lm.out_scores)
                    tk.register_output("runtime", lm.out_runtime)
                    tk.register_output("validation", lm.out_validation)
                scoring.append(lm)
                for n, scale, penalty, policy in selection_grid():
                    condition = f"two-pass-speech-n{n}-scale-{scale:g}-word-penalty{penalty:g}-{policy}"
                    with ExperimentContext(f"open-asr/{group}/{condition}/{name}"):
                        select = runtime.SelectSpeechNbestJob(lm.out_scores, gen_runtime, lm.out_runtime,
                            n=n, scale=scale, word_penalty=penalty, policy=policy, code_version=version)
                        select.add_alias("selection")
                    export_score(group, condition, name, prepared, select.out_hypotheses, select.out_runtime)
                    selections.append(select)
                for n in (10, 50):
                    oracle = runtime.OracleSpeechNbestJob(prepared.out_references, gen.out_candidates, n=n, code_version=version)
                    export_score(group, f"oracle-speech-n{n}", name, prepared, oracle.out_hypotheses)
                    oracles.append(oracle)
    return dict(generation=generation, scoring=scoring, selections=selections, oracles=oracles)


def run():
    data = development()
    return dict(fusion=run_fusion(data), rescoring=run_rescoring(data), pruning=run_pruning(data))
