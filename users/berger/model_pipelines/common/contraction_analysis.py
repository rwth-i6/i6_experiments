"""Reference-free selection and paired reporting for contraction rescoring."""

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from sisyphus import Job, Task

from .contraction_variants import choose

MODES = ('literal', 'contracted', 'expanded', 'best')


class SelectContractionNbestJob(Job):
    def __init__(self, candidates, validation, frozen_hypotheses, *, n, scale, word_penalty, code_version):
        self.candidates, self.validation, self.frozen_hypotheses = candidates, validation, frozen_hypotheses
        self.n, self.scale, self.word_penalty, self.code_version = n, scale, word_penalty, code_version
        self.out_hypotheses = {m: self.output_path(m + '.jsonl') for m in MODES}
        self.out_selections = self.output_path('selections.jsonl')
        self.out_validation = self.output_path('validation.json')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        validation = json.loads(Path(self.validation.get_path()).read_text())
        if not validation['passed'] or not validation['literal_cache_preserved']:
            raise ValueError('Contraction score gate did not pass')
        count = eligible = 0
        with ExitStack() as stack:
            src = stack.enter_context(open(self.candidates.get_path()))
            frozen = stack.enter_context(open(self.frozen_hypotheses.get_path()))
            outputs = {m: stack.enter_context(open(p.get_path(), 'w')) for m, p in self.out_hypotheses.items()}
            ledger = stack.enter_context(open(self.out_selections.get_path(), 'w'))
            for line in src:
                row = json.loads(line)
                old_line = frozen.readline()
                if not old_line:
                    raise ValueError('Missing frozen literal hypothesis')
                old = json.loads(old_line)
                candidates = row['candidates']
                picks = {m: choose(candidates, mode=m, n=self.n, scale=self.scale, word_penalty=self.word_penalty) for m in MODES}
                literal_text = candidates[picks['literal']['index']]['text']
                if row['id'] != old['id'] or literal_text != old['text']:
                    raise ValueError('Literal selection does not exactly reproduce the frozen result')
                touched = any(v['surface'] != c['contraction_scores']['literal']['surface']
                              for c in candidates for v in c['contraction_scores'].values())
                eligible += touched
                for mode, pick in picks.items():
                    candidate = candidates[pick['index']]
                    if not touched and candidate['text'] != literal_text:
                        raise ValueError('Untouched control changed')
                    # Output is always the original AM text, never the LM rendering.
                    outputs[mode].write(json.dumps(dict(id=row['id'], text=candidate['text']))+'\n')
                    rendering = candidate['contraction_scores'][pick['rendering']]
                    pick.update(am_text=candidate['text'], surface=rendering['surface'], rules=rendering['rules'],
                                am_words=len(candidate['text'].split()), changed_output=candidate['text'] != literal_text)
                ledger.write(json.dumps(dict(id=row['id'], eligible=touched, selections=picks))+'\n')
                count += 1
            if frozen.readline() or not count:
                raise ValueError('Frozen/current coverage differs')
        Path(self.out_validation.get_path()).write_text(json.dumps(dict(passed=True, literal_exact_parity=True,
            units=count, eligible_units=eligible, unchanged_control_units=count-eligible), indent=2)+'\n')


class ContractionReportJob(Job):
    def __init__(self, experiments, protocol, *, code_version):
        self.experiments, self.protocol, self.code_version = experiments, protocol, code_version
        self.out_summary = self.output_path('summary.json')
        self.out_report = self.output_path('report.md')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=1, mem=8, time=3))

    def run(self):
        from . import controlled_comparison as cc
        from .convention_audit import jsonl_map, surface_features

        panels = []
        for entry in self.experiments:
            selection = cc.read_json(entry['selection_validation'].get_path())
            gate = cc.read_json(entry['score_validation'].get_path())
            if not selection['passed'] or not gate['passed']:
                raise ValueError('Contraction validation failed')
            modes = list(MODES) + ['am_first_pass', 'speech_standalone']
            units, summaries = {}, {}
            for mode in modes:
                scoring = entry['scores'][mode]
                units[mode] = jsonl_map(scoring['segments'].get_path())
                summaries[mode] = cc.read_json(scoring['summary'].get_path())
                if summaries[mode]['normalizer_revision'] != cc.NORMALIZER:
                    raise ValueError('Contraction comparison normalizer differs')
            ids = set(units['literal'])
            ledger = jsonl_map(entry['selections'].get_path())
            if ledger.keys() != ids or any(u.keys() != ids for u in units.values()):
                raise ValueError('Contraction comparison coverage differs')
            grid = {}
            for j, mode in enumerate(modes):
                for field in ('total', 'ref_len', 'ins', 'del', 'sub'):
                    if sum(r['counts'][field] for r in units[mode].values()) != summaries[mode][field]:
                        raise ValueError('Contraction counts/summary mismatch')
                for seq_id, row in units[mode].items():
                    cluster = cc.cluster_id(entry['dataset'], seq_id)
                    grid.setdefault(cluster, np.zeros((len(modes), 2)))[j] += [row['counts']['total'], row['counts']['ref_len']]
            data = np.array(list(grid.values()))
            against_literal = cc.paired_bootstrap(data, [[], [0], [0], [0], [], []])
            against_components = cc.paired_bootstrap(data, [[4, 5]]*4 + [[], []])
            slices = []
            raw_s = jsonl_map(entry['speech_hypotheses'].get_path(), fields=('id', 'text'))
            if raw_s.keys() != ids:
                raise ValueError('Speech baseline must consist of independent dev segments')
            for label, members in (
                ('all', ids), ('eligible_pool', {i for i in ids if ledger[i]['eligible']}),
                ('untouched_pool', {i for i in ids if not ledger[i]['eligible']}),
                ('speech_contraction', {i for i in ids if surface_features(raw_s[i]['text'])['contraction']}),
                ('speech_no_contraction', {i for i in ids if not surface_features(raw_s[i]['text'])['contraction']})):
                if not members:
                    continue
                for mode in MODES:
                    errors = sum(units[mode][i]['counts']['total'] for i in members)
                    words = sum(units[mode][i]['counts']['ref_len'] for i in members)
                    slices.append(dict(slice=label, mode=mode, units=len(members), wer=100*errors/words,
                        corrected_units=sum(units[mode][i]['counts']['total'] < units['literal'][i]['counts']['total'] for i in members),
                        worsened_units=sum(units[mode][i]['counts']['total'] > units['literal'][i]['counts']['total'] for i in members)))
            rows = []
            for j, mode in enumerate(modes):
                row = dict(mode=mode, counts=summaries[mode], wer=against_literal[j]['wer'])
                if j < 4:
                    row.update(delta_literal=against_literal[j].get('delta', 0),
                               ci_literal=against_literal[j].get('delta_ci', [0, 0]),
                               delta_components=against_components[j]['delta'], ci_components=against_components[j]['delta_ci'])
                rows.append(row)
            renderings = {m: dict(Counter(ledger[i]['selections'][m]['rendering'] for i in ids)) for m in MODES}
            changed = {m: sum(ledger[i]['selections'][m]['changed_output'] for i in ids) for m in MODES}
            panels.append(dict(model=entry['model'], dataset=entry['dataset'], settings=entry['settings'], rows=rows,
                               slices=slices, clusters=len(grid), score_gate=gate, selection_gate=selection,
                               winning_renderings=renderings, changed_outputs=changed,
                               extra_runtime=cc.read_json(entry['extra_runtime'].get_path()),
                               sources={m: {k: dict(path=p.get_path(), sha256=cc.sha256(p.get_path())) for k,p in v.items()}
                                        for m,v in entry['scores'].items()}))
        result = dict(protocol_sha256=cc.sha256(self.protocol.get_path()), panels=panels,
                      note='Development-only fixed-settings comparison. AM strings/costs, retained first-pass LM and AM-word rewards unchanged. '
                           'Best = minimum whole-string Speech NLL over three fixed styles; no exhaustive alias search or probability marginalization. '
                           'Intervals: 10,000 paired source-cluster replicates, seed 20261103, unadjusted. Slices are descriptive/output-conditioned.')
        Path(self.out_summary.get_path()).write_text(json.dumps(result, indent=2)+'\n')
        lines = ['# Contraction variants in Speech-Qwen rescoring', '', result['note'], '',
                 "Ambiguous 's/'d, possessives, numbers and hyphens remain unchanged. Output is always the original AM candidate. "
                 'Speech-Qwen 1.7B and the selected two-pass scales/formatting are held fixed. AM-only first-pass controls '
                 'are specific to each n-best generator, not independently selected beam-search controls.', '']
        for panel in panels:
            lines += [f"## {panel['model']} / {panel['dataset']}", '',
                      f"Settings: {panel['settings']}; {panel['clusters']} clusters. Literal parity passed. "
                      f"Eligible pool units: {panel['selection_gate']['eligible_units']}/{panel['selection_gate']['units']}.", '',
                      '| Mode | WER | I / D / S | Δ vs literal [95% CI] | Δ vs better standalone [95% CI] | Changed outputs |',
                      '|---|---:|---:|---:|---:|---:|']
            for row in panel['rows']:
                c = row['counts']
                fmt = lambda d, ci: f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
                dl = fmt(row['delta_literal'], row['ci_literal']) if 'ci_literal' in row else '—'
                dc = fmt(row['delta_components'], row['ci_components']) if 'ci_components' in row else '—'
                lines.append(f"| {row['mode']} | {row['wer']:.3f} | {c['ins']} / {c['del']} / {c['sub']} | {dl} | {dc} | "
                             f"{panel['changed_outputs'].get(row['mode'], '—')} |")
            lines += ['', '| Slice | Mode | Units | WER | Improved units vs literal | Worse units vs literal |',
                      '|---|---|---:|---:|---:|---:|']
            for row in panel['slices']:
                lines.append(f"| {row['slice']} | {row['mode']} | {row['units']} | {row['wer']:.3f} | "
                             f"{row['corrected_units']} | {row['worsened_units']} |")
            lines += ['', f"Additional rescoring RTF: {panel['extra_runtime']['rtf']:.3f}. "
                      'This excludes the cached first pass and original Speech scoring, and includes literal parity checks. '
                      'It is not an end-to-end runtime comparison. Winning LM-rendering counts: '
                      + str(panel['winning_renderings']), '']
        Path(self.out_report.get_path()).write_text('\n'.join(lines)+'\n')
