# Soft pack launch — 2026-09-21

## Action
Started manager for `config/sae_4a_soft_pack.py` from setup dir
`/e/project1/spell/wu24/2026-07-13_unsupervised` after activating the sis venv (`black` present).

- Command: `sis --log_level 30 m -io -r config/sae_4a_soft_pack.py` via nohup
- pid: 409858
- log: log/sae_4a_soft_pack.manager.20260921T092744Z.log
- pid file: log/sae_4a_soft_pack.manager.pid

## Verification
- Graph status (console -s -c, get_jobs_by_status): 1 job in `queue`
  (`speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O`),
  186 jobs `waiting` (downstream of the pack, consistent with the 196-job graph).
- Job dir: `work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.MXKoywbfon8O`
  (submit_log.run confirms submission, engine_name='slurm', rqmt cpu=64 mem=256G gpu=4
  gpu_mem=96, sbatch_args -A spell -p booster --exclusive, time=11.5h).
- SLURM id: 1926421. `scontrol show job 1926421`: JobState=PENDING,
  Reason=ReqNodeNotAvail,_Reserved_for_maintenance, ScheduledNodeList=jpbo-078-23,
  StartTime=2026-09-21T17:00:00 (node reservation/maintenance window), NumNodes=1,
  gres/gpu=4, exclusive.
- No `log.run.1` exists yet (job has not started) — training has not begun for any of the
  four arms (sf_20 / soft_20 / soft_20_s1 / softshuf_20); nothing to report per-arm yet.
- Manager log tail: only one benign WARNING (abspath inside work dir, `vad_port.py`) —
  no errors, no hash mismatch, no config refusal.

## Result
DONE — manager alive and driving the graph, pack job submitted and queued on SLURM,
pending only a node-reservation/maintenance window (start ETA 17:00). Left the manager
running per instructions (must stay alive ~4h). Did not touch the lexlat profile manager
(3972993) or its job.
