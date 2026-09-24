# Review -- the SECOND packed node (PackedEmcTrainJob.wK3hCW0JdJ6G), 2026-09-15

Verdict: **PASS_WITH_CONCERNS**. The delta is what the dispatch says: four byte-identical arms
(bt_a / bt_b / bt_c / lam3_ct) in one 4-GPU allocation, every read wired to the packed checkpoints,
nothing else moved. I found no path to a wrong number. The two concerns are operational couplings,
neither blocking.

Scope: commit `806ea11` (two NEW files: `.../configs/config_sae_4a_s3b_pack_v2.py`,
`src/speech_llm/sae/emc/test_pack_config2.py`; `git show --name-status` shows nothing else) plus
the untracked `config/sae_4a_s3b_pack2.py`. Uncommitted `config_sae_1g_v1.py` /
`config_sae_3e1_d6_swap_cont_v1.py` are another stage's and reach none of these graphs.

## Verified myself (not read off the implementer report)

**(1) Byte-identity, independently rebuilt.** My own script (scratchpad `verify_pack2.py`) built the
pack with `config_sae_4a_s3b_pack_v2.build()` and the single-arm side with
`config_sae_4a_s3b_rate_v1.build()` after setting `R.ACTIVE_ARMS = G.PACK_ARMS` in memory (no
tracked file touched), then took a RAW unified diff (no normalisation) of
`pack.finalized_config(tag).write()` against `ReturnnTrainingJob.returnn_config.write()`:

| arm | single-arm id (== dispatch) | diff |
|---|---|---|
| bt_a | `ReturnnTrainingJob.6G8CpcswqlTR` | 5 diff lines, all the `model =` line |
| bt_b | `NSxRSjyBWp5g` | idem |
| bt_c | `UjJpp4iQ4kkT` | idem |
| lam3_ct | `DhsiFoHwBP5l` | idem |

Each reconstructed single-arm id equals the dispatched id, so the reference is the reviewed arm.
Also equal per arm: epoch key set 1..8, checkpoint basenames (`epoch.008.pt`),
`learning_rate_file = learning_rates`, and `run_cmd` == `ReturnnTrainingJob._get_run_cmd()` up to
each job's own config path. The written configs carry the intended distinct knobs: bt_a
`lam_bt 0.1 / "full"`, bt_b `0.3 / "full"`, bt_c `0.3 / "output_only"` (all with
`bt_ramp_epochs 4`, `bt_batch_sents 128`, `bt_text_path = TextToPhonemeJob.THKMON3k9LJQ/phon.txt.gz`
and `torch_batching = bt_interleaved_batching`), lam3_ct `lam_content 0.3 / content_k 64 /
content_layer 1` with `MfccCodesHdfJob.ovaQXsbT8s3E` on both train and CV. All four at
`lam_rate 3.0`, `rate_rho_hz 9.6619373279`. Nothing is a silent no-op.

**(2) Reads, per arm and per sub-epoch.** The graph off `config/sae_4a_s3b_pack2.py` is **284 jobs**
/ 508 registered outputs, of which **473** are under
`exp2025_11_06_speech_llms/librispeech/sae_4a_s3b_pack2/`: 118 per arm (8 sub-epochs x 13 files
= theta.stats + 2 splits x {per.json, per.txt, decode_stats.txt, decode_stats.json, phone_rate,
distinct_strings}, + 2 x 2 derangement files at ep4 and ep8 only, + learning_rates / model /
returnn.config / selection.json / selection.txt / selected_epoch) plus one units stats output.
lam3_ct's set is identical to bt_a's (symmetric difference empty). Against the pack-v1 graph the
new graph ADDS 256 jobs: the pack (1), the three MFCC jobs, and 252 reads
(104 emc_train_jobs = 64 DecodeStats + 32 theta slices + 8 phi slices, 64 posterior dumps,
64 GreedyPer, 16 gap, 4 selection) -- 63 per arm, as v1. None of the 256 is already on disk.
Aliases: 538 in the union graph, **0 duplicates**, 257 new, all under `sae/4a/s3b_pack2*` or
`sae/4a/eval/s3b_pack2_*` (+ the 3 borrowed `sae/4a/s3b_rate/mfcc`), none colliding with an
existing on-disk alias path and none under `s3b_pack/`, `s3b_rate_`, `s3b_bt`. Output prefixes
likewise disjoint from `sae_4a_s3b_pack/`, `sae_4a_s3b_bt/`, `sae_4a_s3b_rate/<tag>/`.
`DF6blPpto23t` and `SeZzGUScxq4x` are absent from the graph; the only ReturnnTrainingJobs in it are
the two S0b inits (`65NNK8Bwxdtd`, `HcXzd6M2eyVZ`), exactly as in v1.

**(3) The time request, against measurement.** `rqmt = {gpu 4, cpu 64, mem 256.0, time 9.9,
gpu_mem 96}` -- reproduced exactly; `arm_time_rqmt = {bt_a/b/c 9.0, lam3_ct 6.0}`, allocation
`max x 1.1` (`pack_jobs.py:465`). Observed in pack v1 (`PackedEmcTrainJob.SeZzGUScxq4x`, FINISHED):
`:meta:epoch_train_time_secs` per sub-epoch, slowest arm `spec_speed` = 296/272/278/269/275/275/
283/279 s (~4.6 min), total RETURNN `elapsed: 0:41:14` = **0.687 h for all 8 sub-epochs**; lam1 /
lam10 / spec all 0.655 h. So 9.9 h is ~14x the observed per-arm need: a 1.5x BT slowdown lands at
~1.03 h, and even a 10x slowdown (6.9 h) fits. The 11.5 h clamp (`settings.py:120`,
`min(11.5, time)`, silent truncation) does not bind at 9.9 h, and could not bind on any plausible
BT cost here. The only cost of the request is backfill priority (a 9.9 h exclusive ask against a
~1 h job).

**(4) Hash census** (`scripts/sae_4a_cons_census.py`, current tree, comment lines excluded):
s3 **98**, phase **1021**, rate **91** with `lam3 = ReturnnTrainingJob.DF6blPpto23t` and
`ACTIVE_ARMS = ('lam3',)`, every other rate tag "not built". Pack v1 rebuilt in a fresh process:
`PackedEmcTrainJob.SeZzGUScxq4x`, 285 jobs, rqmt time 6.6 h -- unmoved. A before/after pair is
unnecessary: the two new files are imported only by the new workspace entry and the new test.

**(5) Workspace entry.** `config/sae_4a_s3b_pack2.py` mirrors `config/sae_4a_s3b_pack.py` line for
line (docstring, single import of the v2 module, `py()`, `run = py`); loading it is what produced
the 284-job graph above.

**(6) Test file.** `test_pack_config2.py` is a real test, not a tautology: it pins the four
single-arm ids, diffs the written configs with only each job's own output prefix normalised,
asserts the exact 473-name output set, asserts nothing lands under the BT / cons / pack-v1
prefixes, and asserts 9.9 <= 11.5. It restores `R.ACTIVE_ARMS` in a `finally`.

## Findings

**F1 -- `config_sae_4a_s3b_pack_v2.py:158`: the 4-GPU pack is gated on three never-run CPU jobs that
only one of its four arms needs.** `rate._content_codes(...)` puts `MfccFeatureJob.Y8INRgLo8Avo`,
`MfccKMeansJob.wejLUflqBxxo` and `MfccCodesHdfJob.ovaQXsbT8s3E` in the graph; none has a job
directory (never built -- the arm was held), and the pack's config embeds the codes HDFs, so the
pack is not runnable until all three finish. `MfccCodesHdfJob` asserts per-utterance frame equality
against the L15 feature HDFs at tolerance 0, and the k-means fit was made bit-reproducible only at
commit `53d691a`. If that chain errors, bt_a / bt_b / bt_c -- which need nothing from it -- never
start either. Concrete cost: a blocked or delayed launch; and the repair is not free, because
`config_sae_4a_s3b_pack_v2.py:134` asserts `len(PACK_ARMS) == GPUS_PER_NODE`, so dropping lam3_ct
needs a config edit and re-hashes the pack and all 252 reads (v1 review F2). Cheap insurance: let
the MFCC chain finish before the pack is queued, and treat its stats outputs as the go/no-go.

**F2 -- carried over unchanged from the v1 review, still true here (`pack_jobs.py:84-88`, `:542-560`).**
An operator's "Clear jobs in error state?" / `-c` renames the whole pack dir and destroys ALL FOUR
arms' checkpoints and `arm.finished` markers -- the preserving recovery is the manual removal of
`error.run.1`. Repairing one arm re-funds the other three and all 252 reads. A permanently failing
arm blocks its siblings' `learning_rates` and therefore their selection job (the per-sub-epoch gate
reads are unaffected). Neither the new config docstring nor `config/sae_4a_s3b_pack2.py` says this;
it belongs in the launch instruction.

## Non-findings I checked and cleared

* Per-arm `time_rqmt` entries other than the max are read by nothing (`pack_jobs.py:465` takes
  `max`); there is no per-arm kill. Harmless here -- lam3_ct is the cheapest arm and the whole
  allocation has ~14x headroom.
* `BT_STEP_TIME_FACTOR = 1.5` and `TIME_RQMT_PER_ARM_HOURS = 6.0` trace to the dispatch and to the
  literal `config_sae_4a_s3b_rate_v1.py:418` passes to `emc_training`; `SHARED_NODE_TIME_FACTOR`,
  `GPUS_PER_NODE` and the per-arm cpu/mem/gpu_mem come from `pack_jobs` / `emc_training`'s own
  signature, not retyped. cpu 16 / mem 64 / gpu_mem 96 per arm confirmed against the single-arm
  jobs' rqmt.
* v2 replaced v1's defensive `getattr(rate, "COMPANION_ARMS", {})` with a direct
  `rate.COMPANION_ARMS.get(tag, {})` (`:219`); the attribute exists and is empty for all four tags,
  and the byte-identity diff covers it.
* `net_args` is not a no-op: `emc_train_jobs.subepoch_reads` passes it to `eval_jobs.posterior_dump`
  (`emc_train_jobs.py:1723`), and only lam3_ct gets a non-None value -- identical to what S3b-R
  passes for that arm.
* No train/eval contamination in the delta: training and CV are the seed-0 1 % holdout of
  train-clean-100; all reads are dev-clean / dev-other; the BT text is T_phi (the unpaired LM text,
  `bt_text_path` above), the content targets are MFCC k-means codes of the TRAIN audio; gold phones
  enter scoring and the gap's eligible-tag list only. Gate numbers come from
  `.../sae_4a_s3b_pack2/<arm>/ep4/<split>/{per.json, phone_rate, derangement_gap.json}`, i.e. the
  pack's own prefix, and `phone_rate` is its own registered output (never `rate_in_band`).
* GPU assignment is the sorted-name bijection bt_a 0 / bt_b 1 / bt_c 2 / lam3_ct 3.
* Concurrency: this graph contains S0b, the shared `UnitsHdfJob` and the S3b-R MFCC chain, so its
  manager must not run beside the S0b / S3 / S3b-R / S3b-C / pack-v1 managers (the config says so).
* BT memory: the sentence pool is 100k phone-id lists and the frame pool is 64 frames/unit on the
  device (`bt_aux.py:136-142`), so 4 x 64 GB is not at risk; v1 measured 13.6 GB rss per arm.

Artifacts (session-local scratchpad): `verify_pack2.{py,log}`, `verify_v1_and_mfcc.py`,
`verify_alias.py`, `verify_v2_only.py`, `cfg.<arm>.{pack,single}`, `census.{s3,phase,rate}.after`.
