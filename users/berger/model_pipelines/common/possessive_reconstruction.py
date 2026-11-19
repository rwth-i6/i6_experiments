"""Development pilot for restoring possessives in existing Granite outputs."""

import json
import re
from collections import Counter
from pathlib import Path

from sisyphus import Job, Task

from .benchmark_scoring import scoring_pairs
from .convention_audit import jsonl_map
from .hf_scoring import normalize_hf_english


MARGINS = (0, 2, 5)
_PATTERN = re.compile(r"(?<![\w'’\-])(?P<owner>[a-z]+)\s+(?P<verb>is)(?=\s+[a-z])", re.I)
_NON_OWNERS = set("i you he she it we they this that these those there here who what where how why when which it".split())
_PREPOSITIONS = set("at in of from with to for on by under after before".split())
_COMPLEMENTS = set("a an the to not too very more most so also only still now already here there why how where whether that".split())


def edits(text):
    for match in _PATTERN.finditer(text):
        if match['owner'].lower() not in _NON_OWNERS:
            yield (match.start(), match.end(), match['verb'], text[:match.start()] + match['owner'] + "'s" + text[match.end():])


def rule_edit(text):
    """Apply the first edit with a prepositional owner and a plausible continuation."""
    for start, end, _, alternative in edits(text):
        left = text[:start].lower().split()
        if left and left[-1] == 'the':
            left.pop()
        next_word = text[end:].strip().lower().split()[0]
        if left and left[-1] in _PREPOSITIONS and next_word not in _COMPLEMENTS:
            return alternative
    return text


class ScorePossessiveEditsJob(Job):
    def __init__(self, dataset, hypotheses, speech_repo, *, policy, batch_size, code_version):
        self.dataset, self.hypotheses, self.speech_repo = dataset, hypotheses, speech_repo
        self.policy, self.batch_size, self.code_version = policy, batch_size, code_version
        self.out_scores = self.output_path('edits.jsonl')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=2, mem=96, gpu=1, gpu_mem=48, time=8,
                                   sbatch_args=['-p', 'gpu_48gb']))

    def run(self):
        import torch
        from datasets import load_from_disk

        from .hf_benchmark import audio_array, check_transformers_version
        from .qwen_speech import QwenSpeechScorer
        from .speech_qwen_round import score_speech_candidates

        check_transformers_version()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        backend = QwenSpeechScorer(self.speech_repo.get_path(), batch_size=self.batch_size)
        dataset = load_from_disk(self.dataset.get_path())
        hypotheses = jsonl_map(self.hypotheses.get_path())
        ids = list(map(str, dataset['id']))
        if len(ids) != len(hypotheses) or set(ids) != hypotheses.keys():
            raise ValueError('Granite outputs and Speech-Qwen audio differ in coverage')
        gate = None
        with open(self.out_scores.get_path(), 'w') as dst:
            for index, seq_id in enumerate(ids):
                original = hypotheses[seq_id]['text']
                alternatives = [dict(text=original)] + [dict(text=e[-1]) for e in edits(original)]
                gain, best = 0., original
                if len(alternatives) > 1:
                    backend.set_audio(backend.prepare(audio_array(dataset[index]['audio'])))
                    if gate is None:
                        gate = backend.validate()
                    scored = score_speech_candidates(backend, alternatives, (self.policy,))
                    winner = min(scored, key=lambda row: row['speech'][self.policy]['nll'])
                    gain = scored[0]['speech'][self.policy]['nll'] - winner['speech'][self.policy]['nll']
                    best = winner['text']
                dst.write(json.dumps(dict(id=seq_id, original=original, best=best, gain=gain,
                                          alternatives=len(alternatives)-1)) + '\n')
        if gate is None:
            raise ValueError('No possessive alternatives were scored')


def aligned_hypothesis(reference, hypothesis):
    from kaldialign import align

    return [h for r, h in align(reference, hypothesis, '*') if r != '*']


def edit_attribution(reference, original, edited):
    """Attribute localized edits with a common normalized reference grid."""
    ref = normalize_hf_english(reference).split()
    before = aligned_hypothesis(ref, normalize_hf_english(original).split())
    after = aligned_hypothesis(ref, normalize_hf_english(edited).split())
    return Counter(
        restored_s=sum(r == 's' and a == 'is' and b == 's' for r, a, b in zip(ref, before, after)),
        damaged_is=sum(r == a == 'is' and b != 'is' for r, a, b in zip(ref, before, after)),
    )


def missing_possessives(references, hypotheses, pool_path, n):
    """Check anchored s→is errors in the old winner against its frozen pool."""
    targets = {}
    for seq_id, row in hypotheses.items():
        ref = normalize_hf_english(references[seq_id]['reference']).split()
        hyp = aligned_hypothesis(ref, normalize_hf_english(row['text']).split())
        sites = [i for i in range(1, len(ref)) if ref[i] == 's' and hyp[i] == 'is' and ref[i-1] == hyp[i-1]]
        if sites:
            targets[seq_id] = (ref, sites)
    counts = Counter(units=len(targets), sites=sum(len(sites) for _, sites in targets.values()))
    with open(pool_path) as src:
        for line in src:
            row = json.loads(line)
            target = targets.pop(row['id'], None)
            if target is None:
                continue
            ref, sites = target
            covered = set()
            for candidate in row['candidates'][:n]:
                hyp = aligned_hypothesis(ref, normalize_hf_english(candidate['text']).split())
                for i in sites:
                    stem = ref[i-1]
                    pair = (hyp[i-1], hyp[i])
                    if pair in ((stem, 's'), (stem+'s', '*'), ('*', stem+'s')):
                        covered.add(i)
            counts['sites_covered'] += len(covered)
            counts['units_with_any_covered_site'] += bool(covered)
    if targets:
        raise ValueError('Missing frozen candidate pools for affected utterances')
    return dict(counts)


class PossessiveReportJob(Job):
    def __init__(self, panels, protocol, *, code_version):
        self.panels, self.protocol, self.code_version = panels, protocol, code_version
        self.out_report = self.output_path('report.md')
        self.out_summary = self.output_path('summary.json')
        self.out_hypotheses = self.output_path('hypotheses', directory=True)

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=1, mem=8, time=2))

    def run(self):
        from kaldialign import edit_distance
        import numpy as np
        from .controlled_comparison import cluster_id, paired_bootstrap, sha256

        panels = {}
        lines = ['# Granite possessive reconstruction', '',
                 'Speech-Qwen edits the existing two-pass winner when the best single edit '
                 'improves full-transcript audio-conditioned NLL by more than the given margin. '
                 'Native + rule uses a fixed text rule with no model rescoring.', '']
        out = Path(self.out_hypotheses.get_path())
        out.mkdir(parents=True, exist_ok=True)
        for dataset, panel in self.panels.items():
            load = lambda key: jsonl_map(panel[key].get_path())
            refs, native, literal, proposed = (load(key) for key in ('references', 'native', 'literal', 'edits'))
            if not (refs.keys() == native.keys() == literal.keys() == proposed.keys()):
                raise ValueError('Reconstruction conditions differ in coverage')
            if any(r['group_id'] != i or r['chunk_index'] != 0 for i, r in refs.items()):
                raise ValueError('Expected independent development utterances')
            if any(row['original'] != literal[i]['text'] for i, row in proposed.items()):
                raise ValueError('Speech-Qwen edits do not match the frozen control')
            variants = {
                'native': {i: r['text'] for i, r in native.items()},
                'native + rule': {i: rule_edit(r['text']) for i, r in native.items()},
                'literal two-pass': {i: r['text'] for i, r in literal.items()},
                **{f'edit margin {margin}': {i: r['best'] if r['gain'] > margin else r['original']
                                             for i, r in proposed.items()} for margin in MARGINS},
            }
            baselines = [[], [0], [], *([[2]] * len(MARGINS))]
            grids, totals, attribution = {}, [], []
            for index, (name, texts) in enumerate(variants.items()):
                changes = Counter()
                counts = Counter()
                rows = [dict(id=i, text=text) for i, text in texts.items()]
                with (out / (dataset + '-' + name.replace(' ', '_') + '.jsonl')).open('w') as dst:
                    for row in rows:
                        dst.write(json.dumps(row) + '\n')
                for pair in scoring_pairs(list(refs.values()), rows):
                    seq_id = pair['id']
                    unit = edit_distance(pair['reference'].split(), pair['hypothesis'].split())
                    counts.update(unit)
                    cluster = cluster_id(dataset, seq_id)
                    grids.setdefault(cluster, np.zeros((len(variants), 2)))[index] += [unit['total'], unit['ref_len']]
                    if baselines[index]:
                        original = variants[list(variants)[baselines[index][0]]][seq_id]
                        if original != texts[seq_id]:
                            changes['edited'] += 1
                            changes.update(edit_attribution(refs[seq_id]['reference'], original, texts[seq_id]))
                totals.append(dict(counts))
                attribution.append(dict(changes))
            stats = paired_bootstrap(np.array(list(grids.values())), baselines)
            coverage = missing_possessives(refs, literal, panel['cached_pool'].get_path(), panel['n'])
            panels[dataset] = dict(pool_coverage=coverage, conditions=[
                dict(condition=name, counts=counts, attribution=changes, **stat)
                for name, counts, changes, stat in zip(variants, totals, attribution, stats)
            ])
            lines.extend([f'## {dataset}', '', f'Frozen pool coverage for anchored s→is errors: '
                          f'{coverage.get("sites_covered", 0)} / {coverage["sites"]} sites.', '',
                          '| Condition | WER % | Δ [paired 95% CI] | Edited | Restored s | Damaged correct is |',
                          '|---|---:|---:|---:|---:|---:|'])
            for entry in panels[dataset]['conditions']:
                change = entry['attribution']
                interval = '—'
                if 'delta' in entry:
                    lo, hi = entry['delta_ci']
                    interval = f"{entry['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                lines.append(f"| {entry['condition']} | {entry['wer']:.3f} | {interval} | "
                             f"{change.get('edited', 0)} | {change.get('restored_s', 0)} | "
                             f"{change.get('damaged_is', 0)} |")
            lines.append('')
        lines.append('Word attribution uses a common normalized reference alignment. '
                     'Coverage checks the owner and suffix at the affected position in the original pool. '
                     'Intervals are exploratory and unadjusted.')
        Path(self.out_report.get_path()).write_text('\n'.join(lines) + '\n')
        Path(self.out_summary.get_path()).write_text(json.dumps(dict(
            protocol_sha256=sha256(self.protocol.get_path()), panels=panels), indent=2) + '\n')


def run(output_root):
    from sisyphus import tk

    from . import (benchmark_scoring, controlled_comparison, dynamic_llm, hf_benchmark,
                   nbest_rescoring, qwen_speech, speech_qwen_round)
    from .experiment_context import ExperimentContext
    from .job_reports import register_job_report
    from .public_pilots import analysis_version, completed, frozen_protocol, sha
    from . import possessive_reconstruction as implementation

    protocol, protocol_path, digest = frozen_protocol(output_root)
    audit_path = Path(output_root) / 'convention-audit' / digest[:16] / 'summary.json'
    audit = json.loads(audit_path.read_text())
    if audit['protocol_sha256'] != digest:
        raise ValueError('Native Granite audit does not match the frozen protocol')
    native_rows = {r['dataset']: r['files'] for r in audit['sources']
                   if r['family'] == 'Granite CTC | None | native'}
    gpu_version = analysis_version(implementation, qwen_speech, dynamic_llm, hf_benchmark,
                                   speech_qwen_round, nbest_rescoring)
    cpu_version = analysis_version(implementation, controlled_comparison, benchmark_scoring)
    prefix = 'possessive-reconstruction/' + digest[:16]
    panels = {}
    choice = protocol['selected']['Granite CTC | Speech Qwen 1.7B | two-pass | fast']
    for dataset in protocol['dev_datasets']:
        source = choice['dev'][dataset]
        native = native_rows[dataset]
        if (source['reference_sha256'] != native['references']['sha256']
                or sha(source['references']) != source['reference_sha256']
                or sha(native['hypotheses']['path']) != native['hypotheses']['sha256']):
            raise ValueError('Native Granite and frozen two-pass evidence differ')
        selection = next(s['parameters'] for s in source['stages'] if s['job_type'] == 'SelectSpeechNbestJob')
        scoring = next(s['parameters'] for s in source['stages'] if s['job_type'] == 'ScoreSpeechNbestJob')
        literal = completed(source['hypotheses'])
        with ExperimentContext(f'{prefix}/{dataset}'):
            job = ScorePossessiveEditsJob(completed(scoring['dataset']), literal, completed(scoring['speech_repo']),
                                         policy=selection['policy'], batch_size=scoring['batch_size'],
                                         code_version=gpu_version)
            job.add_alias('speech-scoring')
            tk.register_output('edits.jsonl', job.out_scores)
        panels[dataset] = dict(references=completed(source['references']), native=completed(native['hypotheses']['path']),
                               literal=literal, edits=job.out_scores, cached_pool=completed(selection['candidates']),
                               n=selection['n'])
    with ExperimentContext(prefix):
        report = PossessiveReportJob(panels, protocol_path, code_version=cpu_version)
        report.add_alias('report')
        tk.register_output('summary.json', report.out_summary)
        tk.register_output('report.md', report.out_report)
        tk.register_output('hypotheses', report.out_hypotheses)
    register_job_report('reports/asr-lm/possessive-reconstruction.md', report.out_report,
                        title='Granite possessive reconstruction')
    return report
