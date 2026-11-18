"""Development-only rescoring graphs: four extra GPU jobs, no new AM searches."""

from pathlib import Path

from sisyphus import tk

from .experiment_context import ExperimentContext
from .job_reports import register_job_report
from .public_pilots import completed, frozen_protocol, analysis_version, sha


def run(output_root):
    from . import contraction_rescoring as gpu
    from . import contraction_analysis as cpu
    from . import contraction_variants as rules
    from . import controlled_comparison, benchmark_scoring, qwen_speech, dynamic_llm, speech_qwen_round, hf_benchmark, nbest_rescoring
    from .benchmark_scoring import ScoreBenchmarkJob

    protocol, protocol_path, digest = frozen_protocol(output_root)
    gpu_version = analysis_version(gpu, rules, qwen_speech, dynamic_llm, speech_qwen_round, hf_benchmark, nbest_rescoring)
    cpu_version = analysis_version(cpu, rules, controlled_comparison)
    scoring_version = analysis_version(benchmark_scoring)
    prefix = 'contraction-rescoring/' + digest[:16]
    experiments, jobs = [], []
    for model, slug in (('Granite CTC', 'granite'), ('Parakeet RNN-T', 'parakeet')):
        choice = protocol['selected'][model + ' | Speech Qwen 1.7B | two-pass | fast']
        speech_choice = protocol['selected']['Speech Qwen 1.7B | None | standalone | fast']
        for dataset in protocol['dev_datasets']:
            source = choice['dev'][dataset]
            speech_source = speech_choice['dev'][dataset]
            for r in (source, speech_source):
                if r['status'] != 'complete' or sha(r['references']) != r['reference_sha256'] or sha(r['score']) != r['score_sha256']:
                    raise ValueError('Frozen development evidence incomplete or changed')
            if source['reference_sha256'] != speech_source['reference_sha256']:
                raise ValueError('Rescoring/standalone references differ')
            selection = next(s for s in source['stages'] if s['job_type'] == 'SelectSpeechNbestJob')['parameters']
            scoring = next(s for s in source['stages'] if s['job_type'] == 'ScoreSpeechNbestJob')['parameters']
            generation = next(s for s in source['stages'] if s['job_type'] in ('CompactGraniteNbestJob', 'RnntSpeechNbestJob'))
            settings = {k: selection[k] for k in ('n', 'scale', 'word_penalty', 'policy')}
            with ExperimentContext(f'{prefix}/{slug}/{dataset}'):
                job = gpu.ScoreContractionNbestJob(completed(scoring['dataset']), completed(selection['candidates']),
                    completed(scoring['speech_repo']), policy=settings['policy'], n=settings['n'],
                    batch_size=scoring['batch_size'], code_version=gpu_version)
                job.add_alias('extra-scoring')
                jobs.append(job)
                selected = cpu.SelectContractionNbestJob(job.out_candidates, job.out_validation, completed(source['hypotheses']),
                    **{k: settings[k] for k in ('n', 'scale', 'word_penalty')}, code_version=cpu_version)
                selected.add_alias('selection')
                tk.register_output('candidates.jsonl', job.out_candidates)
                tk.register_output('score-validation.json', job.out_validation)
                tk.register_output('extra-runtime.json', job.out_runtime)
                tk.register_output('selections.jsonl', selected.out_selections)
                tk.register_output('selection-validation.json', selected.out_validation)
                scores = {}
                for mode in cpu.MODES:
                    with ExperimentContext(mode):
                        score = ScoreBenchmarkJob(completed(source['references']), selected.out_hypotheses[mode], code_version=scoring_version)
                        score.add_alias('scoring')
                        tk.register_output('hypotheses.jsonl', selected.out_hypotheses[mode])
                        tk.register_output('summary.json', score.out_summary)
                        scores[mode] = dict(summary=score.out_summary, segments=score.out_segments)
                first = Path(generation['job']) / 'output' / ('first_pass.jsonl' if slug == 'granite' else 'hypotheses.jsonl')
                with ExperimentContext('am-first-pass'):
                    score = ScoreBenchmarkJob(completed(source['references']), completed(first), code_version=scoring_version)
                    score.add_alias('scoring')
                    scores['am_first_pass'] = dict(summary=score.out_summary, segments=score.out_segments)
                scores['speech_standalone'] = dict(summary=completed(speech_source['score']), segments=completed(speech_source['segments']))
                experiments.append(dict(model=model, dataset=dataset, settings=settings, scores=scores,
                    selection_validation=selected.out_validation, score_validation=job.out_validation,
                    selections=selected.out_selections, extra_runtime=job.out_runtime,
                    speech_hypotheses=completed(speech_source['hypotheses'])))
    with ExperimentContext(prefix):
        report = cpu.ContractionReportJob(experiments, protocol_path, code_version=cpu_version)
        report.add_alias('report')
        tk.register_output('summary.json', report.out_summary)
        tk.register_output('report.md', report.out_report)
    register_job_report('reports/asr-lm/contraction-rescoring.md', report.out_report, title='Contraction rescoring')
    return dict(scoring=jobs, report=report)
