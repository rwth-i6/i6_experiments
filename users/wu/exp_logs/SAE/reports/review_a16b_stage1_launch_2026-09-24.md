# Review: A16 (b) stage-1 key-search launch (2026-09-24)

Verdict: BLOCK. Do not launch commits 89502349 / 37a2baf3 / 2f1573bb as they stand.

The launch would hang, so the node slot would be lost:
- Eight null-corpus runs, decipher_context_s01 to s08, start outside the rate band.
- Their start repair can never reach the band.
- The repair loop has no deadline.
- The parent process waits on them until Slurm kills the job at 4 h.
- search.json is never written, so KeySearchSelectJob never runs and there is no stage-1 selection.

I ran a bounded CPU replication of this path on the login node. It is described below.

The review was static and bounded. I submitted nothing, ran no manager and ran no full search.

## Findings, most severe first

### 1. BLOCK: the null context-cluster starts hang in repair_to_band
Code involved:
- key_search.py:719-738, which has no deadline check;
- key_search_jobs.py:188, the call;
- key_search_jobs.py:389-391, where cf.wait has no timeout.

Mechanism:
- The job clusters the run-permuted null corpus by context features (key_search_jobs.py:367-368).
- The resulting Ward partition is degenerate:
  - one class holds 321 of the 500 units and 70.4% of the frames;
  - 37 classes are singletons;
  - class sizes are min 1, median 1, max 321.
- Every one of the 8 null decipher_context starts is VOID on the train side, at 5.665 to 5.683 Hz against the 5.80 floor. All other 292 starts are in band, on both corpora.
- The class-level rate ceiling sits below the floor:
  - an injective map of the 40 classes (SIL on the class with the fewest frames) gives 5.682 Hz;
  - a no-SIL map gives 5.682 Hz.
  - The reason: the 321-unit class is forced onto one symbol, so its consecutive runs merge.

Replication:
- I replicated the job's repair exactly for null decipher_context_s01:
  - same labels;
  - same start, random_class_key(40, 2, 1);
  - same rng seed [1, 2, 1, 0, 1];
  - same strictly-closer accept rule;
  - 1 thread, which is THREADS in the job.
- Result: 3905 proposals in 722 s, 68 accepted.
  - The rate went from 5.6645 to 5.6937 Hz.
  - The last 3000 proposals added only 0.0004 Hz.
  - Evaluate cost: 3.76 s for the big class and 0.100 s for a small class, about 0.19 s per proposal on average.
- The loop's only exit is `assert prop < 1_000_000`. That is about 52 h away, far past the 4 h Slurm limit.

What happens at launch:
- The other runs finish, and their runs/*.json files are written.
- The parent blocks in cf.wait on the 8 hung futures.
- Slurm kills the job at 4 h, before search.json is written. The deadline at 3 h 35 min only reaches anneal(), which these runs never get to.
- KeySearchSelectJob depends on search.json, so it and the agreement job never run.
- A retry repeats the hang, because the seeds are deterministic.

Evidence:
- scratchpad start_rates.log (every phase-1 start rate, both corpora);
- scratchpad null_repair.py / null_repair.log.

Also note:
- Simply capping the proposals with the existing assert does not fix this. A raising run lands in failed_runs, and the job then raises at key_search_jobs.py:437-438 after writing search.json. The job is still in error, so selection is still blocked.
- The fix has to let the job finish. Options:
  - a repair that returns unrepaired and records the run as void or skipped without raising;
  - skipping void class starts;
  - a unit-level repair fallback.
- Which of these to use is a design decision for the coordinator. The spec says "the same search on the destroyed corpus", and the null context clustering is degenerate by construction, because run-permuting randomises context.
- The implementer's statement "No void starts were seen on the real or run-permuted corpora" (impl report, line 128) rests on a few random and argmax starts only. It does not cover the cluster starts.

### 2. Low: a worker death escapes the error handling (key_search_jobs.py:411)
- If a worker process dies (OOM or a signal), the pool breaks.
- The loop records the finished futures as failed. But the next phase-2 `ex.submit` raises BrokenProcessPool outside the try block, so search.json is not written.
- This is unlikely here: the estimated memory is about 284 x 2 GB RSS (shared through copy-on-write) against about 858 GiB.

### 3. Low: selection does not dedupe identical keys (key_search_jobs.py:467-472)
- If a full run and its warm twin are both truncated at their start, which happens after the deadline, they carry the same final key.
- Both can then be selected, which spends two stage-2 arms on one key.
- This only happens under heavy truncation. The expected critical path of about 1.75 to 2 h against the 3 h 35 min deadline makes it unlikely once finding 1 is fixed.

## Checks that passed

### 1. Label-freeness
- The held-out side is used only after the search, and only for void and selection.
- Gold enters only KeyAgreementReportJob.
- The relabel pick is the best of 8 by train J.
- The argmax keys come from the A10/A11 posteriors, and A11 was picked by lowest held-out S.
- The run-permuted null uses no labels.

### 2. Objective and moves match stage 0
- Same J terms (unit_key.j_terms / _j_pair).
- Same prior (RtzbESkOedsT) and duration prior (ReQtJKYpZgsN: d_min 2, D 25/50, overlong split).
- alpha 1e-3, band [5.80, 14.49], rate = 50 x non-SIL tokens / original frames.
- Moves are exact. The implementer's exact test compares incremental moves against a full recount.

### 3. Starts
- Per corpus, phase 1 has 150 runs:
  - 64 random;
  - 7 argmax x 2 schedules;
  - 16 cluster decipherments (8 centroid + 8 context);
  - 56 relabel decipherments.
- Phase 2 has 46 runs:
  - 16 cluster keys x 2 schedules;
  - 7 relabel picks x 2 schedules.
- Seeds are distinct, and each warm run shares its full twin's seed.
- Train and held-out sets are disjoint: 28254 and 260 utterances.

### 4. Robustness (apart from findings 1 and 2)
- Result files are written atomically (tmp file, then os.replace).
- The deadline and truncation logic in anneal() keeps the best key.
- MAX_WORKERS is 284.
- The numba layer is TBB and was checked to be fork-safe.
- The ICM phase is capped at 50 sweeps.
- The only unbounded loop is repair_to_band (finding 1).

### 5. Routing
- The job goes to gpupack as a whole node, le4h bucket, time 240 min.
- The select and agreement jobs run on the login node (short engine).
- No live manager config (em, a14, a17, keyinit) references KeySearchJob. keyinit shares only the finished GoldUnitKeyJob.
- There is therefore no overlap with managers 4111121, 2080167, 1096118 or 1192448.

### 6. settings.py and hashes
- settings.py was last modified 00:47, before these commits. JOB_AUTO_CLEANUP=True is unchanged.
- The stage-0 KeySearch/KeyFloor hashes are unchanged in graph_search.

### 7. Shim
- config/sae_4a_lexlat_v2_keysearch_s1.py already exists (143 B, 09:33) and imports config_sae_4a_lexlat_v2_keysearch_s1_v1.py.
- No new write to config/ is needed.
- Once finding 1 is fixed and re-reviewed, launch with the dispatch's form using `-r config/sae_4a_lexlat_v2_keysearch_s1.py`.
- The KeySearchJob id did not change across 37a2baf3 and 2f1573bb (it stayed AzM1NoHpOnFJ), so a code fix alone will not create a new job directory. After the fix, make sure no errored directory from an earlier attempt exists at that id before launching.

## Not verified
- Timings under full-node contention.
- The class-level global maximum over non-injective maps. The hill climb found 5.694 at most, which is what the job's own procedure would reach.
