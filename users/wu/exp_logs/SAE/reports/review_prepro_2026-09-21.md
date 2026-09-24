# Code review — SAE 4a prepro N = 20 pack (d92109b, f184df4), 2026-09-21

Status: **APPROVE_WITH_AMENDMENTS**. Read-only review; nothing edited or launched.
Scope: `src/speech_llm/sae/emc/trimmed_audio_data.py`, `test_trimmed_audio_data.py`,
`configs/config_sae_4a_prepro_pack_v1.py` against `config_sae_4a_budget_pack_v1.py`,
`config_sae_4a_budget_v1.py`, `blankfree_budget_jobs.py`, `blankfree_train_jobs.py`,
`blankfree_data.py` (the bed), `blankfree_eval_jobs.py`, `vad_port.py`, `build_av_states.py`,
`encoders/wav2vec2.py`, `init_jobs.py`, `pack_jobs.py` and RETURNN itself.

## Findings

### F1 (amendment) — the band does NOT move the batch order; it is an init band
`config_sae_4a_prepro_pack_v1.py:68` and `:165-166` state that `random_seed` moves "phi's init AND
the batch/shuffle order", and the module doc sells `CTRL_S1_SEED` as "the full run-to-run spread".
Half of that is wrong. `random_seed` does reach RETURNN (`returnn/torch/engine.py:1238` at model
creation and `:309` per epoch, `rf.set_random_seed` -> `torch/frontend/_backend.py:60`
`torch.random.manual_seed`), so phi's init and every training RNG draw move. The DATA order does
not: the train dataset runs `seq_ordering="laplace:.1000"` (`blankfree_train_jobs.py:244`), whose
seed is `Dataset._get_random_seed_for_epoch` on top of `random_seed_offset`, and
`returnn/datasets/basic.py:265-291` returns 0 unless `torch_distributed` / horovod / the env var is
set — none of which this config writes. `ctrl_20` and `ctrl_20_s1` therefore see the SAME sequence
order and the same batches; the band is theta-init + phi-init + RNG only.
Consequence for the gate: the ctrl-vs-ctrl spread under-states the true run-to-run spread, so a
prepro-vs-ctrl paired delta that lands OUTSIDE it is not thereby content. Amend the wording in the
config and in `SAE_4A_prepro.md`'s amendment 1 to "initialisation band", or add the data-order knob
explicitly if the wider band is wanted (it would need a `random_seed_offset`-style key, not
`random_seed`).

### F2 (amendment) — the fidelity statistic has no extraction-path null
`trimmed_audio_data.py:494-507` checks only the QUANTIZER call: banked states -> this module's
`assign` -> banked units, exact, once per shard. Nothing checks that this job's own `encode()` on
the UNTRIMMED waveform reproduces the banked states/units. The pre-funding number the pack is
funded on — bed-vs-new unit agreement, `:679` / `:698` — therefore mixes (a) the waveform cut,
(b) any encoder/numerics difference between this job and the banked `AvStatesJob` dump, and (c) the
`raw_index` mapping error near splices. The decision rule ("near 1.00 = empty treatment, do not
fund") and the smoke anecdote 0.738 both read (a) off a number that also carries (b) and (c).
The extraction path looks equivalent on inspection (both batch 1, `no_grad`, fp32 compute ->
fp16 store, same wrapper, per-utterance zero-mean/unit-var on the valid samples;
`build_av_states.py:96-106` vs `trimmed_audio_data.py:431-437`; `encoders/wav2vec2.py:107-116`,
`:152-165`), so the null is expected to be ~1.00 — which is exactly why it is worth one cheap
measurement rather than an argument. Amendment: in the dev-other job, for the first ~50 utterances
also encode the FULL waveform and report agreement of its units against the banked units, beside
the trimmed agreement. Cost: seconds of GPU on a 15-min job.

### F3 (note, read-side) — the agreement denominator includes frames the bed dropped
`trimmed_audio_data.py:566-569`: `matched = banked[raw_index] == units_new` over all T' trimmed
frames, i.e. a trimmed frame whose mapped original frame the bed's OR mask DISCARDED still counts
in the denominator. That is a defensible convention (it is the new stream's own frames) but it is
not "agreement on the frames both streams keep", and the funding rule is stated as a bare number.
State the convention beside the number when the pack is funded.

## What was checked and is sound

1. **ctrl_20 = the budget bed at N = 20.** Same bed object `attrib._control()`, same streams
   `base["vad"]` (`:438`, `:456-466`), same unbiased priorshuf prior (`:452`), `model_args_delta`
   and `bt_kwargs` both `{}` and asserted (`:304-309`), same builder call with the same arguments as
   `config_sae_4a_budget_pack_v1.py:135-153` (prepro `:313-330`; the only structural difference is
   the dropped `out[.../bt_kwargs]` bookkeeping entry, which nothing reads). Batch size, max_seqs,
   optimiser, grad clip are the bed's constants (`blankfree_train_jobs.py:248-258`), and a sub-epoch
   is `EMC_PARTITION_EPOCH = 4` (`blankfree_train_jobs.py:243`, `emc_train_jobs.py:254`) —
   INDEPENDENT of N, so a sub-epoch at N = 20 is the same ~7.1k utterances and the same step count
   as ctrl_50's, and the 601 s/sub-epoch figure carries over. Only `num_epochs`, the kept set and
   the two schedule lists differ.
2. **The N = 20 schedule matches the intent.** Derived from `blankfree_budget_jobs.py:118-157` by
   hand: w = ceil(0.10*20) = 2, h = floor(0.6*20) = 12, peak 1e-4, floor 0.1 ->
   lr = [1e-5, 1e-4 x 11, 8.875e-5, 7.75e-5, 6.625e-5, 5.5e-5, 4.375e-5, 3.25e-5, 2.125e-5, 1e-5];
   tau (`:96-115`, m = 4) = [8.0, 5.0397, 3.1748, 2.0 x 17]. Exactly the dispatch's shape (anneal 4
   sub-epochs 8 -> 2, warmup 2, hold to 12, linear decay to 20 ending at the warmup start value) and
   exactly what `_schedules` asserts (`config:194-213`). `KEEP_EPOCHS = (1, 4, 10, 20)` (`:130`) is
   local because `budget.KEEP_EPOCHS` has no N = 20 entry; it is the plan's own set.
3. **ctrl_20 vs ctrl_20_s1.** `flat_seed` reaches the init (`init_jobs.py:410-413`, `:428`
   `torch.manual_seed(self.seed)`) and is in the job hash; `net_args` is the bed's own
   `blankfree_train_jobs.NET_ARGS` (`config:459` vs `config_sae_4a_blankfree_v1.py:52,84`), so the
   seeded init differs from the bed's in the seed alone. `random_seed` is written only for that arm
   and asserted absent otherwise (`config:332-335`); it is a real RETURNN key (see F1). No other
   difference between the two arms.
4. **prepro_20 vs ctrl_20.** The only delta is the stream object (`config:438`, `:456-466`, `:491`):
   `_TrimmedStreams` exposes the bed's four attribute names, and both the training config and the
   per-epoch reads index `streams[ARMS[arm][2]]`, so PER, decode stats and the derangement gap all
   consume the ARM's own feats/units/orig_length. CV inside training is the arm's own TRAIN streams
   with the cv segment file, as in the budget pack (`:319-321` vs budget `:141-143`). PER is
   sequence-level greedy + SIL-strip + edit distance and never touches `raw_index`
   (`blankfree_eval_jobs.py:75-96`); the gap takes raw hyps, the arm's units and the frozen eta.
   `phone_rate_original_hz = 50 * phones / sum(orig_length)` (`blankfree_eval_jobs.py:106`) and
   `orig_length` is the UNTRIMMED 50 Hz count in both arms (`trimmed_audio_data.py:561`, asserted
   against the banked raw totals `:740-744`), so the rate clause has the identical denominator in
   both arms. eta is a per-TAG frozen table (`sae_emc.py:573-580`), not frame-indexed, so the frame
   count change cannot corrupt it. HDF field parity with the bed: same writer, dims 1024/500/1/1,
   same int32/float16 dtypes, same per-utterance insert pattern (`trimmed_audio_data.py:477-482`,
   `:558-561` vs `blankfree_data.py:140-161`). A dropped utterance cannot silently shrink an
   evaluation: the PER job asserts the full 2864/2703 tag set.
5. **Data-job fidelity.** RAW 10 ms labels drive the cut (`:510`, `:531`), never the 50 Hz mask;
   `speech_segments` (`:159-184`) is the fairseq rule as documented (stride 160, 0->1 opens,
   1->0 closes, open tail to the last sample, no margin/merge, empty list = keep whole);
   `aggregate_labels_to_silence` (`:253-268`) is byte-for-byte `vad_port.rvad_silence`'s rule
   (`vad_port.py:39-43`) and is asserted against the real function on the first utterance of each
   shard (`:513-521`); the OR-mask totals are asserted exactly against 781,130 / 831,372 /
   15,427,853 (`:128`, `:734-739`); `sum(orig_length)` against 919,980 / 968,057 / 18,088,388
   (`:740-744`); utterance counts against 2864 / 2703 / 28,539 (`:730-733`); T' < 2 is a recorded
   offender and a hard STOP, not a silent skip (`:542-551`, `:600-605`, `:725-729`); units come from
   the fp16-ROUNDED states (`:547`, `:553`); the quantizer is the frozen `FWpGhC941JMi` and the
   per-utt/global standardisation branch follows the pickle's own flag (`:396-397`); rVADfast is the
   bed's own instance at threshold 0.4 (`:460` vs `blankfree_data.py:105`); the front-end
   normalisation acts on the TRIMMED waveform per utterance (`encoders/wav2vec2.py:107-116`, `:152`);
   layer 15 = `hidden_states[15]` through the same wrapper the banked dump used
   (`build_av_states.py:96-106`); the encoder runs ONE utterance at a time on both sides, so there is
   no padding effect to compare. `raw_index` uses the amended centre 320 t' + 200 (`:113-117`,
   `:227`) — the Design section's "+160" is superseded by amendment 4, and the code follows the
   amendment.
6. **Pre-funding statistics.** All of them are accumulated per shard and rendered in `summary.txt`
   (`:677-722`): overall agreement, the splice-distance bins, k-means distortion new vs bed, unit
   entropy and dead units, OR-vs-raw delta, T'/T, segments per utterance, T' < 2 count, plus the
   per-shard quantizer agreement. `py_dev_other()` (`config:521-530`) registers the dev-other
   instance alone, and that job (`TrimmedAudioBlankfreeDataJob.0IOLr6hZnYWj`, matching the dispatch)
   is on disk and running. The bin edges are a declared rendering choice (`:139-143`) and the
   edge-free overall number is printed beside them.
7. **Beyond the delta / leakage / resources.** No extra difference found between the arms. No label
   touches training or selection: gold phones enter only the PER and gap readers; the prior is the
   unpaired text prior; `attrib._priorshuf_prior` is the banked pair and is asserted not to be the
   rate module's prior (`config:453`). `raw_index` is emitted but consumed by nothing in this graph
   (diagnostics only) — documented, not a defect. Pack rqmt: gpu 4, time 11.5 h asserted
   (`config:477`); the three arms run CONCURRENTLY, one per GPU (`pack_jobs.py:91-99`), so the
   allocation carries one arm's clock, 20 x 601 s = 3.34 h (x1.1 shared-node margin = 3.7 h), and
   even a serial reading (10.0 h) fits the 11.5 h slot. The data job asks 1 GPU / 24 GB / 4 h per
   shard (`trimmed_audio_data.py:357`).

## CANNOT_TELL

* Byte-identity of `speech_segments` / `concat_segments` with fairseq `vads.py:80-92` and
  `remove_silence.py:44-51`: I did not fetch upstream. The rule as implemented matches the rule as
  quoted in the Design and in the module doc, and the unit tests cover the open tail, the
  all-silence case and single-frame segments.
* The claimed job hashes (pack `YQszIGUOm7Sh`, train `qb4o6dlW3urA`, dev-clean `hxIx0ItTvx15`) and
  the "no banked id moved" census were not re-run here; only `0IOLr6hZnYWj` was confirmed on disk.
