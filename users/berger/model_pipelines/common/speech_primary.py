"""Speech-Qwen beam proposals reranked with full-utterance CTC or RNN-T likelihoods."""

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sisyphus import Job, Task, tk

from .benchmark_scoring import scoring_pairs
from .convention_audit import jsonl_map
from .hf_scoring import normalize_hf_english


WEIGHTS = (0., .03, .1, .3, 1.)


def rnnt_log_probability(blank, emit):
    """Sum legal paths. Emit stays at the current frame, blank advances time."""
    incoming = np.full(blank.shape[1], -np.inf)
    incoming[0] = 0
    for blank_frame, emit_frame in zip(blank, emit, strict=True):
        offsets = np.r_[0., np.cumsum(emit_frame, dtype=np.float64)]
        incoming = np.logaddexp.accumulate(incoming - offsets) + offsets + blank_frame
    return float(incoming[-1])


def acoustic_targets(adapter, text):
    if adapter.family == 'ctc':
        surface = normalize_hf_english(text)
    else:
        # Parakeet emits internal apostrophes, so preserve its contraction
        # spelling while removing sentence punctuation and capitalization.
        surface = text.lower().replace('’', "'")
        surface = re.sub(r"[^\w'\s]", ' ', surface)
        surface = re.sub(r"(?<!\w)'|'(?!\w)", '', surface)
        surface = ' '.join(surface.split())
    ids = adapter.tokenizer.encode(surface, add_special_tokens=False)
    forbidden = set(adapter.tokenizer.all_special_ids) | {adapter.blank}
    if forbidden.intersection(ids) or adapter.detokenize(ids).strip() != surface:
        return None
    return ids


class SpeechPrimaryPoolJob(Job):
    def __init__(self, dataset, speech_repo, baseline, *, beam, length_penalty, max_tokens, version):
        self.dataset, self.speech_repo, self.baseline = dataset, speech_repo, baseline
        self.beam, self.length_penalty, self.max_tokens, self.version = beam, length_penalty, max_tokens, version
        self.out_pool = self.output_path('pool.jsonl')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=2, mem=48, gpu=1, gpu_mem=48, time=16, sbatch_args=['-p', 'gpu_48gb']))

    def run(self):
        import torch
        from datasets import load_from_disk
        from .hf_benchmark import audio_array, check_transformers_version
        from .qwen_speech import QwenSpeechScorer

        check_transformers_version()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        scorer = QwenSpeechScorer(self.speech_repo.get_path(), batch_size=8)
        baseline = jsonl_map(self.baseline.get_path(), fields=('text',))
        seen = set()
        with torch.inference_mode(), open(self.out_pool.get_path(), 'w') as out:
            for sample in load_from_disk(self.dataset.get_path()):
                seq_id = str(sample['id'])
                inputs = scorer.prepare(audio_array(sample['audio']))
                result = scorer.model.generate(
                    **inputs, num_beams=self.beam, num_return_sequences=self.beam,
                    max_new_tokens=self.max_tokens, do_sample=False, length_penalty=self.length_penalty,
                    return_dict_in_generate=True, output_scores=True,
                )
                candidates = {}
                for ids, score in zip(result.sequences[:, inputs['input_ids'].shape[1]:].tolist(),
                                      result.sequences_scores.tolist(), strict=True):
                    # Beam outputs are padded after EOS. Retain the actual turn terminator for decoding.
                    for index, token in enumerate(ids):
                        if token in scorer.eos_ids:
                            ids = ids[:index+1]
                            break
                    text = scorer.processor.decode(ids, return_format='transcription_only')
                    if not math.isfinite(score):
                        raise ValueError(f'Nonfinite Speech-Qwen beam score for {seq_id}')
                    candidates.setdefault(text, dict(text=text, primary_cost=-score))
                if seq_id in seen or next(iter(candidates)) != baseline[seq_id]['text']:
                    raise ValueError(f'Speech-Qwen top-beam parity failed for {seq_id}')
                seen.add(seq_id)
                out.write(json.dumps(dict(id=seq_id, candidates=list(candidates.values())), ensure_ascii=False)+'\n')
        if seen != baseline.keys():
            raise ValueError('Speech-Qwen candidate coverage differs from frozen baseline')


def score_rnnt(adapter, ids, *, validate=False):
    import torch

    tokens = torch.tensor([[adapter.blank, *ids]], device=adapter.device)
    predictors = adapter.model.decoder(tokens, cache=None)[0]
    encoder = adapter.encoder[0, :adapter.num_frames]
    blank_columns, emit_columns = [], []
    for start in range(0, len(ids)+1, 8):
        hidden = predictors[start:start+8]
        width, frames = len(hidden), len(encoder)
        logits = adapter.model.joint(
            decoder_hidden_states=hidden.repeat_interleave(frames, dim=0),
            encoder_hidden_states=encoder.repeat(width, 1),
        )
        probabilities = logits.float().log_softmax(-1).reshape(width, frames, -1)
        if validate and start == 0:
            contexts = [(0, tuple(ids[:i])) for i in range(width)]
            expected = torch.tensor([-row[0][adapter.blank] for row in adapter.score_contexts(contexts)],
                                    device=adapter.device)
            torch.testing.assert_close(probabilities[:, 0, adapter.blank], expected, atol=0.003, rtol=0.001)
        blank_columns.extend(probabilities[:, :, adapter.blank].cpu().numpy())
        for i in range(start, min(start+width, len(ids))):
            emit_columns.append(probabilities[i-start, :, ids[i]].cpu().numpy())
    blank = np.stack(blank_columns, axis=1)
    emit = np.stack(emit_columns, axis=1) if emit_columns else blank[:, :0]
    return -rnnt_log_probability(blank, emit)


class SpeechPrimaryAcousticJob(Job):
    def __init__(self, dataset, pool, repo, *, family, version):
        self.dataset, self.pool, self.repo, self.family, self.version = dataset, pool, repo, family, version
        self.out_scores = self.output_path('scores.jsonl')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=2, mem=48, gpu=1, gpu_mem=48, time=16, sbatch_args=['-p', 'gpu_48gb']))

    def run(self):
        import torch
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, check_transformers_version

        check_transformers_version()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        adapter = HfBenchmarkAdapter(self.repo.get_path(), self.family)
        pool = jsonl_map(self.pool.get_path())
        seen = set()
        gate = False
        with torch.inference_mode(), open(self.out_scores.get_path(), 'w') as out:
            for sample in load_from_disk(self.dataset.get_path()):
                seq_id = str(sample['id'])
                if seq_id in seen:
                    raise ValueError(f'Duplicate audio ID {seq_id}')
                seen.add(seq_id)
                inputs = adapter.inputs(audio_array(sample['audio']))
                if self.family == 'ctc':
                    probs = adapter.ctc_logits(inputs).float().log_softmax(-1)[:, None, :]
                else:
                    adapter.encode_inputs(inputs)
                candidates = pool[seq_id]['candidates']
                cache = {}
                for candidate in candidates:
                    ids = acoustic_targets(adapter, candidate['text'])
                    cost = math.inf
                    if ids is not None:
                        key = tuple(ids)
                        if key not in cache:
                            if self.family == 'ctc':
                                labels = torch.tensor(ids, dtype=torch.long, device=adapter.device)
                                cache[key] = torch.nn.functional.ctc_loss(
                                    probs, labels, [len(probs)], [len(ids)], blank=adapter.blank,
                                    reduction='sum', zero_infinity=False).item()
                            else:
                                cache[key] = score_rnnt(adapter, ids, validate=not gate)
                                gate = True
                        cost = cache[key]
                    candidate['am_cost'] = cost if math.isfinite(cost) else None
                out.write(json.dumps(dict(id=seq_id, candidates=candidates), ensure_ascii=False)+'\n')
        if seen != pool.keys() or (self.family == 'rnnt' and not gate):
            raise ValueError('Incomplete acoustic likelihood coverage or RNN-T validation')


def rerank(candidates, weight):
    # Preserve the entire pool. If any candidate cannot be acoustically scored,
    # leave the utterance at Speech-Qwen beam-8 top hypothesis for all acoustic weights.
    if not weight or any(c['am_cost'] is None for c in candidates):
        return candidates[0]
    return min(candidates, key=lambda c: c['primary_cost'] + weight*c['am_cost'])


class SpeechPrimaryReportJob(Job):
    def __init__(self, panels, *, weights, version):
        self.panels, self.weights, self.version = panels, weights, version
        self.out_report = self.output_path('report.md')
        self.out_summary = self.output_path('summary.json')

    def tasks(self):
        yield Task('run', rqmt=dict(cpu=1, mem=8, time=3))

    def run(self):
        from kaldialign import edit_distance
        from .controlled_comparison import cluster_id, paired_bootstrap

        results = {}
        lines = ['# Speech-Qwen primary rescoring', '',
                 'Development-only. Beam-8 Speech-Qwen 1.7B proposes transcripts. '
                 'AM costs sum over alignments using HF-normalized text for Granite and lowercase text with internal apostrophes for Parakeet. '
                 'Published hypotheses retain the original Speech-Qwen wording. '
                 'An utterance with any unscorable AM candidate keeps the top Speech-Qwen hypothesis. '
                 'The oracle selects by reference errors within the same pool.', '']
        for dataset, panel in self.panels.items():
            refs = jsonl_map(panel['references'].get_path())
            pool = jsonl_map(panel['pool'].get_path())
            if any(r['group_id'] != i or r['chunk_index'] != 0 for i, r in refs.items()):
                raise ValueError('Expected independent development scoring units')
            variants = {'Speech-Qwen beam-8 top hypothesis': {i: r['candidates'][0]['text'] for i, r in pool.items()}}
            oracle = {}
            for i, row in pool.items():
                candidates = [dict(id=i, text=c['text']) for c in row['candidates']]
                def errors(hyp):
                    pair = next(scoring_pairs([refs[i]], [hyp]))
                    return edit_distance(pair['reference'].split(), pair['hypothesis'].split())['total']
                oracle[i] = min(candidates, key=errors)['text']
            variants['Pool oracle'] = oracle
            fallback = {}
            for family, path in panel['acoustic'].items():
                scored = jsonl_map(path.get_path())
                if scored.keys() != pool.keys():
                    raise ValueError('Acoustic scores and pool differ in coverage')
                fallback[family] = sum(any(c['am_cost'] is None for c in r['candidates']) for r in scored.values())
                for w in self.weights:
                    if w:
                        variants[f'{family} weight {w:g}'] = {i: rerank(r['candidates'], w)['text'] for i, r in scored.items()}
            totals, grids = [], {}
            for column, texts in enumerate(variants.values()):
                counts = Counter()
                for pair in scoring_pairs(list(refs.values()), [dict(id=i,text=t) for i,t in texts.items()]):
                    unit = edit_distance(pair['reference'].split(), pair['hypothesis'].split())
                    counts.update({k: unit[k] for k in ('total', 'ref_len', 'ins', 'del', 'sub')})
                    cluster = cluster_id(dataset, pair['id'])
                    grids.setdefault(cluster, np.zeros((len(variants), 2)))[column] += [unit['total'], unit['ref_len']]
                totals.append(dict(counts))
            expected = json.loads(Path(panel['baseline_score'].get_path()).read_text())
            if any(expected[k] != value for k, value in totals[0].items()):
                raise ValueError('Frozen Speech-Qwen baseline WER differs')
            stats = paired_bootstrap(np.array(list(grids.values())), [[]]+[[0]]*(len(variants)-1))
            rows = [dict(condition=name, counts=counts, **stat) for name, counts, stat in zip(variants, totals, stats)]
            controls = {name: json.loads(Path(path.get_path()).read_text())['wer_percent']
                        for name, path in panel['controls'].items()}
            results[dataset] = dict(rows=rows, fallback_utterances=fallback, am_controls=controls,
                                    mean_pool_size=sum(len(r['candidates']) for r in pool.values())/len(pool))
            control_text = ', '.join(f'{name} {wer:.3f}%' for name, wer in controls.items())
            lines += [f'## {dataset}', '', f'Frozen AM controls are {control_text}.',
                      f'AM fallback utterances {fallback}.', '',
                      '| Condition | WER % | Δ vs Speech-Qwen [paired 95% CI] |', '|---|---:|---:|']
            for row in rows:
                ci = row.get('delta_ci')
                change = f"{row['delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else '—'
                lines.append(f"| {row['condition']} | {row['wer']:.3f} | {change} |")
            lines.append('')
        Path(self.out_summary.get_path()).write_text(json.dumps(results, indent=2)+'\n')
        Path(self.out_report.get_path()).write_text('\n'.join(lines)+'\n')


def run(output_root):
    from . import benchmark_scoring, controlled_comparison, hf_benchmark, qwen_speech, speech_primary
    from .experiment_context import ExperimentContext
    from .job_reports import register_job_report
    from .public_pilots import analysis_version, completed, frozen_protocol, sha

    protocol, _, digest = frozen_protocol(output_root)
    selected = protocol['selected']
    version = analysis_version(speech_primary, hf_benchmark, qwen_speech, benchmark_scoring, controlled_comparison)
    prefix = f'speech-primary/{digest[:16]}'
    panels = {}
    for dataset in protocol['dev_datasets']:
        primary = selected['Speech Qwen 1.7B | None | standalone | fast']['dev'][dataset]
        stage, = [s['parameters'] for s in primary['stages'] if s['job_type'] == 'SpeechQwenBeamJob']
        if sha(primary['references']) != primary['reference_sha256'] or sha(primary['score']) != primary['score_sha256']:
            raise ValueError(f'Frozen Speech-Qwen evidence changed in {dataset}')
        audio = completed(stage['dataset'])
        with ExperimentContext(f'{prefix}/{dataset}'):
            pool = SpeechPrimaryPoolJob(audio, completed(stage['speech_repo']), completed(primary['hypotheses']),
                                        beam=stage['beam'], length_penalty=stage['length_penalty'],
                                        max_tokens=stage.get('max_new_tokens', 500), version=version)
            pool.add_alias('beam-pool')
            tk.register_output('pool.jsonl', pool.out_pool)
        acoustic, controls = {}, {}
        for model, family, job_type in (
            ('Granite CTC', 'ctc', 'CompactGraniteNbestJob'),
            ('Parakeet RNN-T', 'rnnt', 'RnntQwenRecogJob'),
        ):
            am = selected[f'{model} | None | standalone | fast']['dev'][dataset]
            if am['reference_sha256'] != primary['reference_sha256'] or sha(am['score']) != am['score_sha256']:
                raise ValueError(f'Frozen AM control differs in {dataset}')
            repositories = {s['parameters']['repo'] for s in am['stages'] if s['job_type'] == job_type}
            repo, = repositories
            with ExperimentContext(f'{prefix}/{dataset}/{family}'):
                job = SpeechPrimaryAcousticJob(audio, pool.out_pool, completed(repo), family=family, version=version)
                job.add_alias('acoustic-scoring')
                tk.register_output('scores.jsonl', job.out_scores)
                acoustic[family] = job.out_scores
            controls[model] = completed(am['score'])
        panels[dataset] = dict(pool=pool.out_pool, acoustic=acoustic, controls=controls,
                               references=completed(primary['references']), baseline_score=completed(primary['score']))
    with ExperimentContext(prefix):
        report = SpeechPrimaryReportJob(panels, weights=WEIGHTS, version=version)
        report.add_alias('report')
        tk.register_output('summary.json', report.out_summary)
        tk.register_output('report.md', report.out_report)
    register_job_report('reports/asr-lm/speech-primary.md', report.out_report, title='Speech-Qwen primary rescoring')
    return report
