"""Additional Speech-Qwen scores on frozen pools, with cached-literal parity gate."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sisyphus import Job, Task

from .contraction_variants import variants


def plans(candidates, policy):
    from .speech_qwen_round import speech_surface
    result, cache = [], {}
    for c in candidates:
        literal = speech_surface(c['text'], policy)
        original = c['speech'][policy]
        costs = original['token_costs']
        if not costs or not np.isfinite(costs).all() or not np.isfinite(original['nll']) or abs(sum(costs) - original['nll']) > 1e-3:
            raise ValueError('Invalid cached literal score')
        if not np.isfinite([c['am_cost'], c['lm_cost'], c['first_pass_cost']]).all() or abs(c['am_cost'] + c['lm_cost'] - c['first_pass_cost']) > .01:
            raise ValueError('Invalid retained first-pass costs')
        if literal in cache and abs(cache[literal]['nll'] - original['nll']) > 1e-6:
            raise ValueError('Inconsistent duplicate literal scores')
        cache[literal] = original
        result.append({name: dict(surface=speech_surface(v['text'], policy), rules=v['rules'])
                       for name, v in variants(c['text']).items()})
    return result, cache


def check_literal_scores(cache, scored, surfaces):
    maximum = 0.
    for surface in surfaces:
        actual, expected = scored[surface]['token_costs'], cache[surface]['token_costs']
        if len(actual) != len(expected):
            raise ValueError('Cached/current literal tokenization differs')
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-4,
                                   err_msg='Literal score replay failed; inspect software/model parity')
        maximum = max(maximum, max(abs(a-b) for a, b in zip(actual, expected)))
    return maximum


def attach_scores(candidates, candidate_plans, cache, scored):
    # Cached literals always take precedence over newly computed gate scores.
    available = {**scored, **cache}
    output = []
    for c, plan in zip(candidates, candidate_plans):
        renderings = {name: dict(p, **available[p['surface']],
                                 source='cached' if p['surface'] in cache else 'new') for name, p in plan.items()}
        output.append(dict(text=c['text'], am_cost=c['am_cost'], lm_cost=c['lm_cost'],
                           first_pass_cost=c['first_pass_cost'], contraction_scores=renderings))
    return output


class ScoreContractionNbestJob(Job):
    def __init__(self, dataset, cached_candidates, speech_repo, *, policy, n, code_version, batch_size=8, gate_segments=8):
        if n < 1 or batch_size < 1 or gate_segments < 1:
            raise ValueError('Positive pool, batch and literal-gate sizes required')
        self.dataset, self.cached_candidates, self.speech_repo = dataset, cached_candidates, speech_repo
        self.policy, self.n, self.code_version = policy, n, code_version
        self.batch_size, self.gate_segments = batch_size, gate_segments
        self.out_candidates = self.output_path('contraction_candidates.jsonl')
        self.out_runtime = self.output_path('runtime.json')
        self.out_validation = self.output_path('validation.json')
        self.rqmt = dict(cpu=2, mem=96, gpu=1, gpu_mem=48, time=24, sbatch_args=['-p', 'gpu_48gb'])

    def tasks(self):
        yield Task('run', rqmt=self.rqmt)

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
        elapsed = audio_seconds = maximum_difference = 0.
        gated = new_surfaces = reused_surfaces = count = 0
        seen = set()
        backend_validation = None
        with open(self.cached_candidates.get_path()) as src, open(self.out_candidates.get_path(), 'w') as dst:
            for i, sample in enumerate(dataset):
                line = src.readline()
                if not line:
                    raise ValueError('Missing candidate segment')
                row = json.loads(line)
                if row['id'] != str(sample['id']) or row['id'] in seen or not row['candidates']:
                    raise ValueError('Audio/candidate ID coverage mismatch')
                seen.add(row['id'])
                candidates = row['candidates'][:self.n]
                candidate_plans, cache = plans(candidates, self.policy)
                surfaces = list(dict.fromkeys(p['surface'] for plan in candidate_plans for p in plan.values()))
                missing = [s for s in surfaces if s not in cache]
                gated_surfaces = list(cache) if i < self.gate_segments else []
                requested = list(dict.fromkeys(gated_surfaces + missing))
                waveform = audio_array(sample['audio'])
                torch.cuda.synchronize()
                started = time.perf_counter()
                scored = {}
                if requested:
                    backend.set_audio(backend.prepare(waveform))
                    if backend_validation is None:
                        backend_validation = backend.validate()
                    forward = score_speech_candidates(backend, [dict(text=s) for s in requested], ('original',))
                    scored = {surface: r['speech']['original'] for surface, r in zip(requested, forward)}
                    maximum_difference = max(maximum_difference, check_literal_scores(cache, scored, gated_surfaces))
                    gated += bool(gated_surfaces)
                # Reuse cached literal scores EXACTLY, including when a transformed
                # surface equals another literal candidate. No control-score drift.
                output = attach_scores(candidates, candidate_plans, cache, scored)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - started
                audio_seconds += len(waveform)/16000
                new_surfaces += len(missing)
                reused_surfaces += len(surfaces)-len(missing)
                count += 1
                dst.write(json.dumps(dict(id=row['id'], candidates=output), ensure_ascii=False) + '\n')
                dst.flush()
                print(f'{count}/{len(dataset)}: extra rescoring RTF={elapsed/audio_seconds:.3f}', flush=True)
            if src.readline():
                raise ValueError('Extra cached candidate segments')
        if not count or gated != min(self.gate_segments, count) or backend_validation is None:
            raise ValueError('Incomplete literal gate')
        validation = dict(passed=True, backend=backend_validation, gated_segments=gated,
                          max_literal_token_cost_difference=maximum_difference, literal_cache_preserved=True,
                          literal_tolerance=dict(atol=2e-3, rtol=2e-4),
                          new_surfaces=new_surfaces, reused_surfaces=reused_surfaces,
                          input_sha256=hashlib.sha256(Path(self.cached_candidates.get_path()).read_bytes()).hexdigest())
        Path(self.out_validation.get_path()).write_text(json.dumps(validation, indent=2)+'\n')
        Path(self.out_runtime.get_path()).write_text(json.dumps(dict(elapsed_seconds=elapsed, audio_seconds=audio_seconds,
            rtf=elapsed/audio_seconds, num_segments=count, batch_size=self.batch_size, dtype='float32',
            scope='Additional surfaces and literal gate only; cached generation/original scoring excluded'), indent=2)+'\n')
