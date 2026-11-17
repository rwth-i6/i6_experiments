"""Report the paired clean/noisy pilot at frozen development-selected settings."""

import json
from pathlib import Path

import numpy as np
from sisyphus import Job, Task


class NoisePilotReportJob(Job):
    def __init__(self, results, manifest, protocol, *, code_version, bootstrap_samples=10000):
        self.results, self.manifest, self.protocol = results, manifest, protocol
        self.code_version, self.bootstrap_samples = code_version, bootstrap_samples
        self.out_summary = self.output_path("summary.json")
        self.out_report = self.output_path("report.md")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=1))

    def run(self):
        from . import controlled_comparison as cc
        from .convention_audit import jsonl_map

        manifest = cc.read_json(self.manifest.get_path())
        ids = {r["id"] for r in manifest["units"]}
        keys = list(self.results)
        totals, runtimes, counts_by_cluster = [], [], {}
        for key in keys:
            entry = self.results[key]
            summary = cc.read_json(entry["score"].get_path())
            if summary["normalizer_revision"] != cc.NORMALIZER:
                raise ValueError("Noise pilot normalizer mismatch")
            units = jsonl_map(entry["segments"].get_path())
            if units.keys() != ids:
                raise ValueError("Noise conditions differ in scoring-unit coverage")
            for field in ("ins", "del", "sub", "total", "ref_len"):
                if sum(r["counts"][field] for r in units.values()) != summary[field]:
                    raise ValueError("Noise pilot segment/summary mismatch")
            totals.append(summary)
            runtime = cc.read_json(entry["runtime"].get_path())
            if runtime['num_segments'] != len(ids) or not np.isclose(runtime['audio_seconds'], manifest['audio_seconds'], atol=1e-5):
                raise ValueError('Noise pilot runtime/audio coverage mismatch')
            runtimes.append(runtime)
            for seq_id, row in units.items():
                chapter = cc.cluster_id("librispeech-dev-other", seq_id)
                grid = counts_by_cluster.setdefault(chapter, np.zeros((len(keys), 2)))
                grid[len(totals) - 1] += [row["counts"]["total"], row["counts"]["ref_len"]]
        # Resample the SAME source chapters jointly across clean and both SNRs.
        baseline_indices = [[keys.index(key.split('/')[0] + '/parakeet'), keys.index(key.split('/')[0] + '/speech')]
                            if key.endswith('/fusion') else [] for key in keys]
        stats = cc.paired_bootstrap(np.array(list(counts_by_cluster.values())), baseline_indices,
                                    seed=manifest["seed"], samples=self.bootstrap_samples)
        rows = [dict(condition=key, **s, counts=summary, timing=runtime)
                for key, s, summary, runtime in zip(keys, stats, totals, runtimes)]
        result = dict(protocol_sha256=cc.sha256(self.protocol.get_path()), manifest=manifest,
                      normalizer=cc.NORMALIZER, clusters=len(counts_by_cluster), bootstrap_samples=self.bootstrap_samples,
                      intervals="Exploratory unadjusted 95% paired chapter bootstrap; own denominators retained", rows=rows,
                      sources={key: {name: dict(path=p.get_path(), sha256=cc.sha256(p.get_path()))
                                     for name, p in entry.items()} for key, entry in self.results.items()})
        Path(self.out_summary.get_path()).write_text(json.dumps(result, indent=2) + "\n")
        lines = ["# Controlled white-noise pilot", "", f"{len(ids)} development utterances; "
                 f"{len(counts_by_cluster)} chapters; {manifest['audio_seconds']/3600:.2f} hours per condition.", "",
                 "Clean and noisy copies use identical IDs, segmentation and references, shared noise realizations "
                 "and a common peak-safe gain per utterance. SNR is whole-segment RMS without VAD. "
                 "This is additive stationary Gaussian noise, not a benchmark of realistic environmental noise.", "",
                 "All model/search settings are frozen from the clean public protocol. No noise-specific tuning. "
                 "Speech-Qwen is 1.7B, native beam 8 / length penalty 0; Parakeet is the selected AM-only beam control; "
                 "fusion is the selected one-pass condition. Δ is against the better constituent at that SNR.", "",
                 "| Condition | WER % | I / D / S | Δ WER points | 95% CI for Δ | RTF |",
                 "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            s = r['counts']
            ci = r.get('delta_ci')
            delta = f"{r['delta']:+.3f}" if ci else '—'
            interval = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else '—'
            lines.append(f"| {r['condition']} | {r['wer']:.3f} | {s['ins']} / {s['del']} / {s['sub']} | "
                         f"{delta} | {interval} | {r['timing']['rtf']:.3f} |")
        lines += ["", result['intervals'] + ". No pooling across the paired noise conditions. "
                  "Runtime measurements remain hardware-dependent. Inspect both constituent degradation and fusion "
                  "gains before deciding whether a larger noise study is informative."]
        Path(self.out_report.get_path()).write_text('\n'.join(lines) + '\n')
