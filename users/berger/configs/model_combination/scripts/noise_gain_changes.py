"""Paired change in fusion gain under noise, from a completed pilot summary.

Usage: python noise_gain_changes.py /path/to/noise-pilot/summary.json
No inference, Sisyphus imports or source-output modifications.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(path):
    summary = json.loads(Path(path).read_text())
    keys = list(summary['sources'])
    clusters = {}
    for j, key in enumerate(keys):
        source = summary['sources'][key]['segments']
        content = Path(source['path']).read_bytes()
        if hashlib.sha256(content).hexdigest() != source['sha256']:
            raise ValueError('Segment evidence changed')
        for line in content.decode().splitlines():
            row = json.loads(line)
            parts = row['id'].split('-')
            if len(parts) != 3 or not all(p.isdigit() for p in parts):
                raise ValueError('Expected LibriSpeech utterance IDs')
            chapter = '-'.join(parts[:2])
            clusters.setdefault(chapter, np.zeros((len(keys), 2)))[j] += [row['counts']['total'], row['counts']['ref_len']]
    counts = np.array(list(clusters.values()))
    point = 100 * counts.sum(0)[:, 0] / counts.sum(0)[:, 1]
    rng = np.random.default_rng(summary['manifest']['seed'])
    samples, n = summary['bootstrap_samples'], len(counts)
    draws = []
    for start in range(0, samples, 32):
        weights = rng.multinomial(n, np.full(n, 1/n), size=min(32, samples-start))
        totals = (weights @ counts.reshape(n, -1)).reshape(len(weights), len(keys), 2)
        draws.append(100 * totals[:, :, 0] / totals[:, :, 1])
    draws = np.concatenate(draws)

    def delta(values, level):
        p, s, f = [keys.index(level + '/' + model) for model in ('parakeet', 'speech', 'fusion')]
        return values[..., f] - np.minimum(values[..., p], values[..., s])

    contrasts = {}
    for level in ('snr20', 'snr10'):
        difference = delta(draws, level) - delta(draws, 'clean')
        contrasts[level] = dict(delta_pp=float(delta(point, level) - delta(point, 'clean')),
                                ci95=np.quantile(difference, [.025, .975]).tolist())
    return dict(summary_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                seed=summary['manifest']['seed'], chapters=n, bootstrap_samples=samples, contrasts=contrasts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary')
    print(json.dumps(analyze(parser.parse_args().summary), indent=2))
