# VAD job runtime estimate (2026-09-24 22:15)
Job BlankfreeVadHdfJob.RLrgIh6lFv9m (Slurm 4337501, cn-604, start 18:59:31, limit 8 h -> 02:59). At 22:07 used_time 3.13 h, cpu ~100 %, RSS 0.45 GB, output/ empty (still phase 1). log.run.1 stops at 19:20 (monitor line only); usage.run.1 ticks.
## Measured on desktop (i5-13600, one pinned core, OMP/MKL/OPENBLAS=1), probes in scratch
- Phase 1 (iter_ogg_zip_audio + rvad_silence), 150 train utts / 0.535 h audio: decode 10.0 s + VAD 22.4 s = 61 s per audio hour.
- Phase 2 (h5py read + SimpleHDFWriter feats write, 600 dev-clean utts, 221k frames): 32 s per M frames.
## Workload (zip metadata)
train 28539 utts 100.6 h; dev-clean 2703 5.4 h; dev-other 2864 5.1 h; total 111.1 h, ~20M 50 Hz frames; input feature HDFs 41 GB.
## Projection
Desktop-equivalent: phase 1 1.9 h + phase 2 ~0.2-0.3 h = ~2.1 h. Job is already 3.13 h in phase 1, so cn-604 per-core speed is at least 1.7x slower than the desktop (node CPU model not readable: no ssh, Slurm reports no features; node load 65/96). Limit is hit only if slowdown > ~3.8x. Plausible slowdown 2-3x -> total 4.2-6.3 h -> finish ~23:10-01:20. NFS I/O in phase 2 (41 GB read) adds uncertainty.
## Reference
JUPITER SAjz8y1cT06g: used_time 1.75 h (34,106 utts + ~35 GB HDF rewrite), exp_logs/SAE/reports/survey_audio_pipeline_2026-09-20.md.
