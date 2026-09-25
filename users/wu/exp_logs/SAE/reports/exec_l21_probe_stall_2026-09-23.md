# L2-1 probe stall check, 2026-09-23
STATUS: DONE. False stall: the whole graph is finished. No relaunch.
Graph status (console, config/sae_4a_lexlat_v2_em_probe.py): 0 unfinished jobs. Slurm 1971921-1971926 all COMPLETED 0:0, 16:16 elapsed each.
The manager log has no errors (0-byte-equivalent end: warnings only), so the manager exited cleanly.
Finished (finished.tar.gz): PhiFirstProbeTrainingJob TVCw6EcU5ahd F4WuEU8vqmeI (uniform s1/s2), zWqS49iSFTdV G270UY24rWaM (durinit), ivlkWhhMJd53 OAlQmq7yNkQh (durfrz); GenMargSampleJob.b7aFZd9Tse5X.
Reader finished: work/speech_llm/sae/emc/blankfree_phifirst_probe_read/PhiFirstProbeReadJob.RuFm51PHSz4q/output/{report.txt,probe_read.json}
Reader VERDICT: NO WAVE SETTING. All settings FAIL on gain only (projection and rate pass).
Per sub-epoch 1..4, held-out tau=1 NLL/frame (S; lower is better) / genmarg emitted non-SIL rate in Hz over original frames:
durinit s1 5.773/1.36 4.906/6.72 3.968/7.69 3.683/7.83 (gain3->4 0.285)
durinit s2 5.774/1.41 4.916/6.74 3.988/7.50 3.709/7.47 (0.279)
durfrz  s1 5.771/1.31 4.919/6.68 4.017/7.67 3.717/7.59 (0.299)
durfrz  s2 5.773/1.37 4.908/6.76 4.036/7.46 3.739/7.42 (0.296)
uniform s1 5.812/0.98 4.906/4.27 4.074/6.17 3.760/6.53 (0.314)
uniform s2 5.814/0.93 4.919/4.57 4.042/6.34 3.709/6.74 (0.333)
Timing (uniform s1 probe.json): 57 steps/sub-epoch, ~200 s train per sub-epoch (210/199/200/201), sec/step mean 3.44 (median 3.60), GPU util mean 95%, peak alloc 15.8 GiB. Measured wall 0.246 h per restart, projected 0.262 h (all six within 0.2464-0.2468 h measured).
