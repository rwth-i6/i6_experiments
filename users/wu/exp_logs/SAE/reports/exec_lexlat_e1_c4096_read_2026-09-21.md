DONE (read-only report)

C=4096 job fY18YN7LVdAe (Slurm 1923324), dir work/speech_llm/sae/emc/lexlat_train_jobs/LexlatEfficiencyProbeJob.fY18YN7LVdAe

1. Failure class: OOM. error.run.1 is empty (0 bytes). Decisive line in log.run.1:
"OutOfMemoryError: CUDA out of memory. Tried to allocate 22.62 GiB. GPU 0 has a total capacity of 95.00 GiB of which 22.31 GiB is free. Including non-PyTorch memory, this process has 72.68 GiB memory in use. Of the allocated memory 71.23 GiB is allocated by PyTorch..."
Traceback location: lexlat.py line 1453 `_replay` -> line 1256 `_step`, torch.gather with b=114, n_new=4096, m_max=6503.
requested_resources in usage.run.1: gpu_mem 96, mem 64.0, time 4.0.

2. No per-batch timer lines ("[lexlat] step N ...") present in the log at all -- died mid-forward on the first batch, before any timer/partial write. 0 of 9 timed batches completed.

3. No partial.json exists for this job (output/ only has returnn.config). No output/.../sae_4a_lexlat/e1_efficiency*/summary exists for either C (searched output/, none found).

4. No peak-GiB line for C=4096 (crashed before logging one); OOM message itself is the only memory evidence (above).

5. Wall time: start 2026-09-21 07:56:48, end 08:26:31 (Run time 0:29:42, per "Max resources" line).

6. Sibling C=1024 x3LaY6KlB6I4 (Slurm 1923286), read-only, appears to still be running (usage.run.1 timestamp 08:37:50, later than its log's last entry):
- Timer lines: "[lexlat] step 0: T=770 B=114 padded=87780 m_max=3840 867.031 s alloc 38.71 GiB reserved 41.10 GiB"; "[lexlat] step 1: T=689 B=127 padded=87503 m_max=3520 911.564 s alloc 37.91 GiB reserved 41.13 GiB" (2 of planned 9 timed batches done so far, both well under STEP_ABORT_SEC=1200).
- partial.json exists (n_measured=2, n_banked=2, aborted_at=null); peak_allocated_gib 38.71/37.91, peak_reserved_gib 41.10/41.13.
- No summary/efficiency.json yet (run not finished).

recipe/i6_experiments/users/wu/exp_logs/SAE/reports/exec_lexlat_e1_c4096_read_2026-09-21.md
