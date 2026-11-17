"""Small opt-in graphs built only from completed frozen public artifacts.

Do not import the full open_asr experiment package: these graphs require no
downloads, builds or reconstruction of historical recognitions.
"""

import hashlib
import json
from pathlib import Path

from sisyphus import tk

from .experiment_context import ExperimentContext
from .job_reports import register_job_report


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_version(*modules):
    from . import hf_scoring

    paths = [Path(m.__file__) for m in modules] + [Path(hf_scoring.__file__)]
    paths += sorted(Path(hf_scoring.__file__).with_name('hf_normalizer').glob('*.py'))
    return hashlib.sha256(b'\0'.join(p.read_bytes() for p in paths)).hexdigest()


def completed(path):
    return tk.Path(str(Path(path).resolve(strict=True)))


def frozen_protocol(output_root):
    source = Path(output_root).resolve() / "controlled-comparison/selection.json"
    protocol = json.loads(source.read_text())
    if not protocol["ready"]:
        raise ValueError("Public development selection is not frozen")
    return protocol, completed(source), sha(source)


def register_convention_audit(output_root):
    from . import convention_audit

    root = Path(output_root).resolve()
    protocol, _, digest = frozen_protocol(root)
    comparison = root / 'controlled-evaluation' / digest[:16] / 'comparison.json'
    evidence = json.loads(comparison.read_text())
    if evidence['protocol_sha256'] != digest:
        raise ValueError('Frozen evaluation protocol differs')
    families = {m + ' | None | standalone' for m in ('Granite CTC', 'Parakeet RNN-T', 'Speech Qwen 1.7B')}
    families |= {m + ' | Speech Qwen 1.7B | one-pass' for m in ('Granite CTC', 'Parakeet RNN-T')}
    sources = [v['dev'][dataset] for k, v in protocol['selected'].items() if k.endswith(' | fast') and v['family'] in families
               for dataset in protocol['dev_datasets']]
    sources += [r for panel in evidence['comparisons'].values() for r in panel['rows'] if r['family'] in families]
    # Tree lemma orthography can impose capitalization. Native controls are
    # surface-only references, kept separate from the frozen paired comparisons.
    datasets = {r['dataset'] for r in sources}
    native = {}
    for partition in ('dev', 'evaluation'):
        snapshot = json.loads((root / f'controlled-comparison/{partition}-snapshot.json').read_text())
        for r in snapshot['records']:
            expected_group = {'Granite CTC': 'qwen-granite', 'Parakeet RNN-T': 'qwen-parakeet-rnnt'}.get(r['model'])
            if r['dataset'] in datasets and r['group'] == expected_group and r['condition'] == 'native' and r['status'] == 'complete':
                native[r['dataset'], r['model']] = dict(r, family=r['model'] + ' | None | native')
    sources += list(native.values())
    records = []
    for r in sources:
        if r['status'] != 'complete':
            raise ValueError('Convention audit requires completed frozen outputs')
        if not r['family'].endswith(' | native'):
            chosen = protocol['selected'][r['family'] + ' | fast']
            if (r['group'], r['condition']) != (chosen['group'], chosen['condition']):
                raise ValueError('Convention audit found a non-frozen condition')
        if sha(r['score']) != r['score_sha256']:
            raise ValueError('Frozen scoring output changed')
        if sha(r['references']) != r['reference_sha256']:
            raise ValueError('Frozen reference output changed')
        records.append(dict(dataset=r['dataset'], family=r['family'], normalizer=r['normalizer'],
                            **{k: completed(r[k]) for k in ('references', 'hypotheses', 'segments')}))
    with ExperimentContext('convention-audit/' + digest[:16]):
        job = convention_audit.ConventionAuditJob(records, protocol_sha256=digest, code_version=analysis_version(convention_audit))
        job.add_alias('analysis')
        tk.register_output('summary.json', job.out_summary)
        tk.register_output('report.md', job.out_report)
    register_job_report('reports/asr-lm/transcription-conventions.md', job.out_report, title='Transcription conventions')
    return job


def replay(stage, dataset, *, speech_baseline=None):
    """Reuse frozen constructors/versions/configs; only change paired audio inputs."""
    from .rnnt_llm import RnntQwenRecogJob
    from .pretrained_correction import SpeechQwenBeamJob

    classes = {c.__name__: c for c in (RnntQwenRecogJob, SpeechQwenBeamJob)}
    if stage['job_type'] not in classes:
        raise ValueError('Unaudited pilot runtime: ' + stage['job_type'])
    for record in stage.get('files', {}).values():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Frozen search input changed')
    paths = {'auxiliary_repo', 'binary', 'config', 'dataset', 'gate', 'repo', 'speech_baseline', 'speech_repo', 'vocabulary'}
    kwargs = {k: completed(v) if k in paths and isinstance(v, str) else v for k, v in stage['parameters'].items()}
    kwargs['dataset'] = dataset
    if speech_baseline is not None:
        kwargs['speech_baseline'] = speech_baseline
    if stage['job_type'] == 'RnntQwenRecogJob':
        kwargs['resume'] = True
    job = classes[stage['job_type']](**kwargs)
    job.rqmt.update(cpu=2, gpu=1, gpu_mem=48, mem=96, time=6, sbatch_args=['-p', 'gpu_48gb'])
    return job


def register_noise_pilot(output_root, *, count=256, seed=20261117):
    from ...data import noise_pilot as data
    from . import noise_pilot as analysis
    from . import benchmark_scoring, controlled_comparison, convention_audit
    from .benchmark_scoring import ScoreBenchmarkJob

    protocol, path, digest = frozen_protocol(output_root)
    families = {'parakeet': 'Parakeet RNN-T | None | standalone',
                'speech': 'Speech Qwen 1.7B | None | standalone',
                'fusion': 'Parakeet RNN-T | Speech Qwen 1.7B | one-pass'}
    sources = {k: protocol['selected'][f + ' | fast']['dev']['librispeech-dev-other'] for k, f in families.items()}
    datasets = {r['stages'][0]['parameters']['dataset'] for r in sources.values()}
    references = {r['reference_sha256'] for r in sources.values()}
    if len(datasets) != 1 or len(references) != 1:
        raise ValueError('Pilot controls did not use the same original data')
    if any(sha(r['references']) != r['reference_sha256'] for r in sources.values()):
        raise ValueError('Frozen pilot references changed')
    prefix = f'noise-pilot/{digest[:16]}/n{count}-seed{seed}'
    with ExperimentContext(prefix):
        prep = data.PrepareNoisePilotJob(completed(next(iter(datasets))), completed(sources['speech']['references']),
                                         count=count, seed=seed, code_version=sha(data.__file__))
        prep.add_alias('data')
        tk.register_output('manifest.json', prep.out_manifest)
        tk.register_output('references.jsonl', prep.out_references)
    scores, results, recognitions = {}, {}, []
    # Standalone Speech is scored before fusion at EACH noise level.
    for level in prep.levels:
        for label in ('parakeet', 'speech', 'fusion'):
            with ExperimentContext(f'{prefix}/{level}/{label}'):
                r = sources[label]
                if len(r['stages']) != 1:
                    raise ValueError('Expected single-stage frozen pilot condition')
                job = replay(r['stages'][0], prep.out_datasets[level],
                             speech_baseline=scores.get(level + '/speech') if label == 'fusion' else None)
                job.add_alias('recognition')
                # Same scorer/normalizer as the frozen study; only new paired data.
                score = ScoreBenchmarkJob(prep.out_references, job.out_hypotheses,
                                          code_version=analysis_version(benchmark_scoring))
                score.add_alias('scoring')
                scores[level + '/' + label] = score.out_summary
                results[level + '/' + label] = dict(score=score.out_summary, segments=score.out_segments, runtime=job.out_runtime)
                for name, value in [('hypotheses.jsonl', job.out_hypotheses), ('runtime.json', job.out_runtime),
                                    ('score.json', score.out_summary), ('segments.jsonl', score.out_segments)]:
                    tk.register_output(name, value)
                recognitions.append(job)
    with ExperimentContext(prefix):
        report = analysis.NoisePilotReportJob(results, prep.out_manifest, path,
                                              code_version=analysis_version(analysis, controlled_comparison, convention_audit))
        report.add_alias('report')
        tk.register_output('summary.json', report.out_summary)
        tk.register_output('report.md', report.out_report)
    register_job_report('reports/asr-lm/noise-pilot.md', report.out_report, title='Controlled noise pilot')
    return dict(data=prep, recognition=recognitions, report=report)
