# Launch: P0 k2lat V100 probes (2026-09-25)

Status: DONE_WITH_CONCERNS

Pre-check at 16:43: neither probe dir held slurm-*.out or lexlat_k2_ABORT.json. Both commands were submitted exactly as briefed from S.

| arm | Slurm | partition/node | start | state (16:49) | chunk line |
|---|---|---|---|---|---|
| A (cs16) | 4362704 | gpu_32gb / cn-32 | 16:44:01 | RUNNING | `installed (chunk_seqs = 16)` |
| B (cs8) | 4362705 | gpu_32gb / cn-32 | 16:44:01 | FAILED 11:0 after 4:14 | `installed (chunk_seqs = 8)` |

Seed (seed_record.json, identical for A and B): source J epoch.003.pt (internal epoch 3, step 171), presented as epoch 7, trains epoch 8. File sha256s: f9212055...a01037 and a2edc3e7...7f5f43 (the model and opt files, same for both). Both jobs started before J's epoch.004 (~17:05), so no mismatch.

A: epoch 8 started at step 171; HLG loaded on cuda:0 (24.9M states); stability check passed (median 0.0194 nats/frame, 16/16). At 16:49 no train-step line had appeared yet.

B: CUDA OOM during sub-epoch 8, before its first logged train step: "Tried to allocate 118 MiB; total 31.73 GiB, 57 MiB free; process 31.67 GiB in use; 29.96 GiB allocated". No lexlat_k2_ABORT.json was written. Note that 29.96 GiB allocated is already above the 28 GiB G0.K2M gate, with chunk_seqs 8.

Dirs: /work/asr4/hwu/sae_i6_probes/p0_k2lat_v100_ep8_2026-09-25{,_cs8}. Logs: slurm-<id>.out and returnn.log there. Nothing was cancelled, deleted or resubmitted. J was not touched.
