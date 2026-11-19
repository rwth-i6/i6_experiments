"""Evaluate the frozen 2-nat Granite reconstruction on the public test suite."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sisyphus import Job, Task, tk

from . import benchmark_scoring, controlled_comparison, possessive_reconstruction
from .benchmark_scoring import scoring_pairs
from .controlled_comparison import cluster_id, paired_bootstrap
from .convention_audit import jsonl_map
from .experiment_context import ExperimentContext
from .job_reports import register_job_report
from .possessive_reconstruction import ScorePossessiveEditsJob, edit_attribution, rule_edit
from .public_pilots import analysis_version, completed, frozen_protocol, sha


MARGIN = 2
GRANITE = 'Granite CTC | Speech Qwen 1.7B | two-pass'
SPEECH = 'Speech Qwen 1.7B | None | standalone'
CONDITIONS = ('Native Granite', 'Native + text rule', 'Literal two-pass', 'Two-pass + reconstruction',
              'Standalone Speech-Qwen 1.7B')


class ScoreTestPossessiveEditsJob(ScorePossessiveEditsJob):
    def tasks(self):
        # SPGISpeech has 100 hours of audio and runs as a single job.
        yield Task('run', rqmt=dict(cpu=2, mem=96, gpu=1, gpu_mem=48, time=48,
                                   sbatch_args=['-p', 'gpu_48gb']))


def score_panel(dataset, panel, output_dir, *, margin):
    from kaldialign import edit_distance

    refs = jsonl_map(panel['references'].get_path())
    native = jsonl_map(panel['native'].get_path(), fields=('text',))
    literal = jsonl_map(panel['literal'].get_path(), fields=('text',))
    speech = jsonl_map(panel['speech'].get_path(), fields=('text',))
    edits = jsonl_map(panel['edits'].get_path())
    if not (refs.keys() == native.keys() == literal.keys() == speech.keys()):
        raise ValueError(f'Baseline segment coverage differs in {dataset}')
    if not refs.keys() <= edits.keys() or any(edits[i]['original'] != literal[i]['text'] for i in refs):
        raise ValueError(f'Edit scores do not match the frozen two-pass output in {dataset}')
    groups = defaultdict(list)
    for seq_id, ref in refs.items():
        groups[ref['group_id']].append(seq_id)
    for members in groups.values():
        members.sort(key=lambda seq_id: refs[seq_id]['chunk_index'])

    native_text = {i: row['text'] for i, row in native.items()}
    literal_text = {i: row['text'] for i, row in literal.items()}
    variants = [
        native_text,
        {i: rule_edit(value) for i, value in native_text.items()},
        literal_text,
        {i: edits[i]['best'] if edits[i]['gain'] > margin else value for i, value in literal_text.items()},
        {i: row['text'] for i, row in speech.items()},
    ]
    counts, attribution, grids = [], [], {}
    for index, texts in enumerate(variants):
        rows = [dict(id=i, text=value) for i, value in texts.items()]
        if index in (1, 3):
            with (output_dir / f'{dataset}-{index}.jsonl').open('w') as dst:
                for row in rows:
                    dst.write(json.dumps(row, ensure_ascii=False) + '\n')
        totals, changes = Counter(), Counter()
        for pair in scoring_pairs(list(refs.values()), rows):
            unit = edit_distance(pair['reference'].split(), pair['hypothesis'].split())
            totals.update({key: unit[key] for key in ('total', 'ins', 'del', 'sub', 'ref_len')})
            group = pair['id']
            grid = grids.setdefault(cluster_id(dataset, group), np.zeros((len(variants), 2)))
            grid[index] += [unit['total'], unit['ref_len']]
            if index in (1, 3):
                ids = groups[group]
                original = variants[index - 1]
                n_edits = sum(original[i] != texts[i] for i in ids)
                if n_edits:
                    changes['edited_segments'] += n_edits
                    changes.update(edit_attribution(refs[ids[0]]['reference'],
                                                    ' '.join(original[i] for i in ids),
                                                    ' '.join(texts[i] for i in ids)))
        counts.append(dict(totals))
        attribution.append(dict(changes))

    for index, summary_path in panel['controls'].items():
        expected = json.loads(Path(summary_path.get_path()).read_text())
        if any(value != expected[key] for key, value in counts[index].items()):
            raise ValueError(f'Frozen score mismatch in {dataset} for {CONDITIONS[index]}')

    stats = paired_bootstrap(np.array(list(grids.values())), [[], [0], [], [2, 4], []])
    return dict(segments=len(refs), scoring_units=len(groups), clusters=len(grids), conditions=[
        dict(condition=name, counts=total, attribution=changes, wer=stat['wer'],
             comparisons=[dict(baseline=CONDITIONS[b['index']], delta=b['delta'], delta_ci=b['delta_ci'])
                          for b in stat.get('baselines', [])])
        for name, total, changes, stat in zip(CONDITIONS, counts, attribution, stats)
    ])


class PossessiveEvaluationReportJob(Job):
    def __init__(self, panels, frozen_comparison, *, margin, code_version):
        self.panels, self.frozen_comparison, self.margin = panels, frozen_comparison, margin
        self.code_version = code_version
        self.out_report = self.output_path('report.md')
        self.out_summary = self.output_path('summary.json')
        self.out_hypotheses = self.output_path('hypotheses', directory=True)

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=2, mem=16, time=12))

    def run(self):
        directory = Path(self.out_hypotheses.get_path())
        directory.mkdir(parents=True, exist_ok=True)
        lines = ['# Granite possessive reconstruction on public test sets', '',
                 f'The Speech-Qwen score margin is fixed at {self.margin} nats. '
                 'WER includes every scoring unit. Attribution compares each edit condition to '
                 'its corresponding original output on a common normalized reference grid. '
                 'Confidence intervals use 10,000 paired cluster bootstrap replicates. '
                 'The Loquacious total and its source subsets overlap.', '']
        panels = {}
        for dataset, panel in self.panels.items():
            result = score_panel(dataset, panel, directory, margin=self.margin)
            panels[dataset] = result
            lines.extend([f'## {dataset}', '',
                          f'{result["scoring_units"]:,} scoring units in {result["clusters"]:,} clusters.', '',
                          '| Condition | WER % | Δ vs original [95% CI] | Δ vs Speech-Qwen [95% CI] | Edited segments | Restored s | Damaged correct is |',
                          '|---|---:|---:|---:|---:|---:|---:|'])
            for row in result['conditions']:
                comparisons = [f"{c['delta']:+.3f} [{c['delta_ci'][0]:+.3f}, {c['delta_ci'][1]:+.3f}]"
                               for c in row['comparisons']]
                original, speech = (comparisons + ['—', '—'])[:2]
                change = row['attribution']
                lines.append(f"| {row['condition']} | {row['wer']:.3f} | {original} | {speech} | "
                             f"{change.get('edited_segments', 0)} | {change.get('restored_s', 0)} | "
                             f"{change.get('damaged_is', 0)} |")
            lines.append('')
        Path(self.out_report.get_path()).write_text('\n'.join(lines) + '\n')
        comparison = self.frozen_comparison.get_path()
        Path(self.out_summary.get_path()).write_text(json.dumps(dict(
            margin_nats=self.margin, frozen_comparison_sha256=sha(comparison), panels=panels), indent=2) + '\n')


def run(output_root):
    from . import dynamic_llm, hf_benchmark, nbest_rescoring, possessive_evaluation, qwen_speech, speech_qwen_round

    protocol, _, digest = frozen_protocol(output_root)
    root = Path(output_root)
    comparison_path = root / 'controlled-evaluation' / digest[:16] / 'comparison.json'
    comparison = json.loads(comparison_path.read_text())
    audit = json.loads((root / 'convention-audit' / digest[:16] / 'summary.json').read_text())
    if any(record['protocol_sha256'] != digest for record in (comparison, audit)):
        raise ValueError('Public test artifacts use different selection protocols')
    native_sources = {r['dataset']: r['files'] for r in audit['sources']
                      if r['family'] == 'Granite CTC | None | native'}
    gpu_version = analysis_version(possessive_evaluation, possessive_reconstruction, qwen_speech,
                                   dynamic_llm, hf_benchmark, speech_qwen_round, nbest_rescoring)
    cpu_version = analysis_version(possessive_evaluation, possessive_reconstruction,
                                   controlled_comparison, benchmark_scoring)
    prefix = f'possessive-evaluation/{digest[:16]}'
    records = {}
    for panel in comparison['comparisons'].values():
        dataset = panel['dataset']
        if dataset in protocol['dev_datasets']:
            raise ValueError('Expected test data only')
        records[dataset] = {row['family']: row for row in panel['rows']}
    panels, edit_scores = {}, {}
    for dataset, rows in records.items():
        parent = 'loquacious-test' if dataset.startswith('loquacious-test.') else dataset
        literal, speech = rows[GRANITE], rows[SPEECH]
        native = native_sources[dataset]
        ref_hash = literal['reference_sha256']
        if (ref_hash != speech['reference_sha256'] or ref_hash != native['references']['sha256']
                or sha(literal['references']) != ref_hash or sha(speech['references']) != ref_hash
                or sha(native['hypotheses']['path']) != native['hypotheses']['sha256']):
            raise ValueError(f'Frozen reference or hypothesis mismatch in {dataset}')
        for record in (literal, speech):
            if record['status'] != 'complete' or sha(record['score']) != record['score_sha256']:
                raise ValueError(f'Frozen score missing or changed in {dataset}')
        if parent not in edit_scores:
            source = records[parent][GRANITE]
            selection = next(s['parameters'] for s in source['stages'] if s['job_type'] == 'SelectSpeechNbestJob')
            scoring = next(s['parameters'] for s in source['stages'] if s['job_type'] == 'ScoreSpeechNbestJob')
            with ExperimentContext(f'{prefix}/{parent}'):
                job = ScoreTestPossessiveEditsJob(completed(scoring['dataset']), completed(source['hypotheses']),
                                                 completed(scoring['speech_repo']), policy=selection['policy'],
                                                 batch_size=scoring['batch_size'], code_version=gpu_version)
                job.add_alias('speech-scoring')
                tk.register_output('edits.jsonl', job.out_scores)
                edit_scores[parent] = job.out_scores
        panels[dataset] = dict(references=completed(literal['references']), native=completed(native['hypotheses']['path']),
                               literal=completed(literal['hypotheses']), speech=completed(speech['hypotheses']),
                               edits=edit_scores[parent], controls={0: completed(Path(native['segments']['path']).with_name('summary.json')),
                                                                   2: completed(literal['score']), 4: completed(speech['score'])})
    with ExperimentContext(prefix):
        report = PossessiveEvaluationReportJob(panels, completed(comparison_path), margin=MARGIN, code_version=cpu_version)
        report.add_alias('report')
        tk.register_output('summary.json', report.out_summary)
        tk.register_output('report.md', report.out_report)
        tk.register_output('hypotheses', report.out_hypotheses)
    register_job_report('reports/asr-lm/possessive-evaluation.md', report.out_report,
                        title='Granite possessive evaluation')
    return report
