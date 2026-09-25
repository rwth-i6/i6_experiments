# Extract: HLG size clause + k2lat step-1 vs ctrl_20 (2026-09-25)

## 1. Trie: LexiconTrieBuildJob.W0e4no47Crfu
Path: work/i6_experiments/users/wu/experiments/unsupervised_asr/lm/word_lm/LexiconTrieBuildJob.W0e4no47Crfu

IDENTITY lines (from `log.run.1`, lines 49-51, 67; also `output/summary.txt` last "IDENTITY vs banked" line):
```
IDENTITY vs banked: trie_words 182215 (banked 151731) MISMATCH
IDENTITY vs banked: window_word_types 182215 (banked 151731) MISMATCH
IDENTITY vs banked: bigram_types {'in_line': 3414449, 'with_bos': 3448655, 'with_bos_eos': 3510073} (banked 3302936) MISMATCH
IDENTITY vs banked: trie_words MISMATCH, window_word_types MISMATCH, bigram_types MISMATCH
```

`identity_vs_banked` block from `output/build.json`:
```json
{
  "bigram_types": {"expected": 3302936, "got": {"in_line": 3414449, "with_bos": 3448655, "with_bos_eos": 3510073}, "pass": false},
  "trie_words": {"expected": 151731, "got": 182215, "pass": false},
  "window_word_types": {"expected": 151731, "got": 182215, "pass": false}
}
```
(identical block appears at top and root level of build.json)

## 2. HLG job consumed by k2lat_20_ma3000

k2lat job `work/i6_core/returnn/training/ReturnnTrainingJob.jcKXbLMDk4hl` input dir lists:
`i6_experiments_users_wu_experiments_unsupervised_asr_lm_hlg_LexlatHLGBuildJob.avjHv1Xvjyqd`

Job dir (resolved): `/work/asr4/hwu/setups/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/i6_experiments/users/wu/experiments/unsupervised_asr/lm/hlg/LexlatHLGBuildJob.avjHv1Xvjyqd`
(setup-relative: `work/i6_experiments/users/wu/experiments/unsupervised_asr/lm/hlg/LexlatHLGBuildJob.avjHv1Xvjyqd`)

Finished: yes (`finished` and `finished.run.1` present).

From `output/summary.txt` / `output/build.json` (identical to `work/build_attempt.json`):
- H: 42 states, 1681 arcs (blank-free run-collapse, min_frames=1 = ceil(d_min 2 / stride 3))
- L: 1076120 states, 1440630 arcs (182215 pronunciations, max disambig #17, sil_prob 0.5)
- G: 3692293 states, 21216609 arcs (182218 unigrams + 3510073 bigrams + 10623445 trigrams)
- HLG: 24949308 states, 103206474 arcs
- pruning: theta = 0.0 nats (the FULL banked trigram, nothing dropped)
- ladder tried: [0.0]  (build.json `prune_ladder`: [0.0, 0.5, 2.0]; `chosen_theta_nats`: 0.0)
- build: 265.9 s (build.json "seconds": 265.87196765094995), peak RSS 14.5 GiB (build.json "max_rss_gib": 14.544471740722656)
- k2 version: 1.24.4.dev20260924+cuda12.6.torch2.7.1
- Stage table (states/arcs/seconds/max_rss_gib) for L_disambig, G, compose(L,G), connect, remove_epsilon, connect, arc_sort, compose(H,LG), connect, arc_sort — see summary.txt / build.json "stages" array verbatim.
- Escape info: GRAPH PRICED = ESCAPE, escape word `<unk>`, escape_disambig #59, escape price range [-8.553378733884081, -3.123625782321416] nats, dropped_words ["<s>","</s>"], determinized: false.

## 3. k2lat step 1 — ReturnnTrainingJob.jcKXbLMDk4hl

`work/returnn.log` (23 lines total) device line (line 51 in `log.run.1`, corresponding line in returnn.log block):
```
  1/1: cuda:0
       name: Tesla V100-SXM3-32GB
       total_memory: 31.7GB
```
No explicit "Using gpu device" string was found in this job's logs (grep for that exact string returned nothing in either `work/returnn.log` or `log.run.1`); the device info instead appears as the "Available CUDA devices" block naming `Tesla V100-SXM3-32GB` (cuda:0).

"ep 1 train, step 0" line: MISSING (searched: `work/returnn.log`, `log.run.1` — no "ep 1 train" or "train, step" string present yet).

As of the last read (log.run.1 last line timestamped 2026-09-25, 13:13:12), the job is still in the data-loading/HDF-copying phase (`parsing file .../BlankfreeVadHdfJob.../orig_length.train.shard*.hdf`, `LOG: destination: /var/tmp/hwu/.../feats.train.shard0.hdf`), with repeated `ERROR: cannot receive: timed out [_readAll]` / `ERROR: no connection to master` lines. Training (epoch/step logging) has not started; no step-0 or later step line exists yet.

Latest logged line (log.run.1, last line): `MEMORY: total (main 607733, 2026-09-25, 13:13:12, 5 procs): pss=436.1MB uss=426.6MB`

Seqs/frames of step 0: MISSING (no step-0 line exists).

## 4. ctrl_20 step 1 — ReturnnTrainingJob.GiT88bxzoZbZ (for comparison)

`log.run.1` line 670 (also `work/returnn.log` line 207), full "ep 1 train, step 0" line:
```
ep 1 train, step 0, l_tau -0.352, agg 1.789, rate 0.069, blankfree_l_tau_per_frame -0.352, blankfree_agg_kl_unigram 0.403, blankfree_agg_kl_bigram 1.386, blankfree_reverse_per_frame -7.209, blankfree_prior_per_token -5.637, blankfree_phone_rate_original_hz 12.400, blankfree_phone_rate_retained_hz 15.091, blankfree_expected_phone_rate_hz 11.874, blankfree_expected_tokens 63.854, blankfree_z_zero_frac 0.000e+00, blankfree_rate_fd_check 7.701e-06, blankfree_rate_dp_calls 2.000, blankfree_temperature 8.000, blankfree_frames_per_sec 0.000e+00, num_seqs 128, max_size:time:var-unk:features 330, max_size:time:var-unk:units 330, max_size:time:var-unk:original_length 1, 37.787 sec/step, elapsed 0:00:46, exp. remaining 0:41:07, complete 1.87%
```
num_seqs (step 0): 128. No separate "frames" key logged at step 0; `max_size:time:var-unk:features`/`units` = 330 is the max-size-in-batch value, not a total frame count. Preceding line (log.run.1 line 669): `ep 1 train num_seqs: 7066` (epoch-level seq count, not step-0-specific).

Device line: `log.run.1:270` / `work/returnn.log:51`: `Using gpu device 0: NVIDIA L40S`

Latest step line in this log (log.run.1:2296): `ep 15 train, step 3, l_tau 1.861, agg 1.446, rate 0.016, ... complete 7.30%` (job has progressed to epoch 15; no timestamp field embedded in the step line itself beyond "elapsed 0:03:14"). `finished` marker present only as `finished.create_files.1`; no top-level `finished`/`finished.run.1` file found in this job dir listing.

## Comparison note (no conclusion drawn)
k2lat (jcKXbLMDk4hl) has not yet produced an "ep 1 train, step 0" line — it is still loading VAD/feature HDF shards as of the last log read. ctrl_20 (GiT88bxzoZbZ)'s step 0 line is reported above verbatim for when k2lat's step 0 becomes available.
