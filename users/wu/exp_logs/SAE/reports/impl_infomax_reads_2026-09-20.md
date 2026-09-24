# Implementer: the InfoMax design review's two per-checkpoint reads (amendments 1, 5, 7) -- 2026-09-20

Second InfoMax implementer round. Status **DONE_WITH_CONCERNS** (one convention the phase file must
pin, item 5 below). Nothing launched, no manager started or restarted, no budget config, model,
train step, `infomax.py`, `consistency.py`, `lattice.py` or `agg.py` touched.

## 1. Item 1: which route, and why

**Route A (a CPU reader on the existing dump), not a new GPU forward.** The registered blank-free
epoch reads ALREADY dump the recognizer's log posteriors for every kept epoch and dev split:
`blankfree_eval_jobs.epoch_reads` calls `eval_jobs.posterior_dump`, a `ReturnnForwardJobV2` writing
`posteriors.hdf` as `[T_out, 40]` float32 **log** probabilities, one row per VALID output frame
(`PosteriorHdfCallback`; `BlankfreeGreedyPerJob` asserts `len(q) == (retained + 2) // 3`). That dump
is EVAL mode by construction -- `eval_jobs.posterior_forward_step` runs under `torch.no_grad()` in
RETURNN's forward engine, i.e. dropout off and BatchNorm on running statistics -- which is exactly
the mode amendment 8 pins. So the entropy of amendment 1 is a read of a file the graph already
produces; a second GPU forward of the same checkpoint would have been duplicate compute.

Consequence for the deliverable: items 1(b) (symbol-usage entropy) and 1(c) (top-symbol shares) are
delivered by the item-2 reader off the decode, as the brief's own fallback clause foresees; the
entropy reader delivers 1(a) plus its spread.

## 2. What was written (all NEW files)

* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/blankfree_infomax_read_jobs.py` (NEW)
  * `PosteriorEntropyJob` (CPU, in-process `run()`): mean per-output-frame entropy in NATS, pooled
    over the valid output frames of the split, from `posteriors.hdf`. Convention printed in the job
    docstring and rendered into the json/txt (pre-registration lives with the code); it is the
    `as_error` convention of `infomax.frame_entropy_loss`, so the eval-mode read and the arms'
    training column are ONE quantity in two modes. Asserts every row is a normalized log
    distribution (max |logsumexp| < 1e-3) and that the dump holds the split's full utterance count.
  * `SymbolUsageNullJob` (CPU): symbol-usage entropy in bits beside the arm's own
    length-and-unigram-matched chance null. The usage entropy is
    `analysis/emc_hyp_inspect.entropy_bits` (the primitive behind the banked 4.3-4.5 / gold 4.8
    bits) and the null is `analysis/emc_null_sdic.analyse_decode` -- the attrib step 6 reader that
    `blankfree_null_jobs.NullAdjustedEditCountsJob` calls, at its own `draws = 5`, `seed = 0`.
    Neither is re-implemented. `analyse_decode` also asserts the read reproduces the decode's banked
    `per.json` S/D/I/N exactly. Json carries `{symbol_entropy_bits, per, null_per, per_minus_null}`
    plus both inventories, the top-5 symbol shares, the decode's rate, and the two PRE-REGISTERED
    rules verbatim (amendment 5's 3-bit low-inventory FAIL, amendment 7's 0.05 band-exit margin).
  * A new module on purpose: three blank-free nodes re-import the blank-free tree at every 11.5 h
    resume, and `blankfree_eval_jobs` / `blankfree_null_jobs` classes are banked.
* `.../sae/emc/test_blankfree_infomax_reads.py` (NEW) -- 9 tests, all passing (section 4).
* `.../configs/config_sae_4a_infomax_pack_v1.py` -- +79 lines, 0 deletions (purely additive):
  `READ_SPLIT = "dev-other"`, `READ_ARMS = tuple(ARM_SPEC) + (BASELINE_ARM,)`,
  `register_infomax_reads(...)`, its call in `py()` (via a `reads_of` closure that reaches `ctrl_50`
  through the existing `_baseline_node()` graph), and the docstring paragraph describing the reads.

## 3. Registration (item 3)

5 arms (`ent_50`, `entaug_50`, `aug_50`, `aughi_50`, `ctrl_50`) x 5 kept epochs (1, 4, 10, 25, 50) x
2 readers = **50 new jobs, 100 new registered outputs**, dev-other only:

* `output/.../sae_4a_infomax_pack/<arm>/ep<k>/reads/posterior_entropy.{json,txt}`
* `output/.../sae_4a_infomax_pack/<arm>/ep<k>/reads/symbol_usage_null.{json,txt}`
* aliases `sae/4a/infomax_pack/<arm>/ep<k>/reads/{entropy,usage_null}` (50 aliases, verified).

`ctrl_50`'s ten reads are NEW jobs on that arm's ALREADY BANKED dump and decode, filed under this
phase's own tree: no output of `config_sae_4a_budget_pack_v1` is re-registered, no job of it is
re-created, and the dumps were already dependencies of the phase's paired reads, so this funds no
new GPU work. Verified: `ctrl_50/ep1/reads/entropy` consumes
`ReturnnForwardJobV2.FI29gVlt2cCT/output/posteriors.hdf`, the finished ep1 dump on disk.

## 4. Checks run and results

1. New tests -- `pytest .../test_blankfree_infomax_reads.py`: **9 passed**. Entropy of a hand-made
   posterior in closed form (uniform over 4 outputs and over 40, pooled); the SAME rows through
   `infomax.frame_entropy_loss`'s `as_error` monitor agree to 1e-9 (the read IS the training
   monitor's quantity, padding excluded); an unnormalized or truncated dump stops the read; symbol
   entropy of a hand-made decode uniform over 4 symbols = **2.000 bits exactly**, collapsed onto one
   symbol = 0 bits, a symbol outside the inventory raises; the null reader end to end on a 3-line
   fixture decode written to disk (banked S/D/I/N reproduced, `null_per` equals the mean over the 5
   draws computed independently, `per_minus_null = per - null_per`); the pre-registered rules are
   verbatim in the job docstrings.
2. Real-data wiring check (login node, CPU, seconds; NOT a banked result -- the registered jobs bank
   these numbers): the entropy reader on the three finished `ctrl_50` dev-other dumps reads
   **3.0412 / 2.4058 / 0.3825 nats per output frame at ep1 / ep4 / ep10** over 2864 utterances and
   261,295 output frames each, max |logsumexp| 3e-7 (the dump is exactly normalized), 0.3-0.5 s per
   dump. Same direction and size as the code review's 16-utterance measurement (3.25 / 2.71 / 0.30),
   which is the expected difference between 16 utterances and the whole split.
   The usage/null reader on `ctrl_50`'s banked dev-other decodes: ep1 usage 2.824 bits (40 outputs)
   / 3.031 bits (39 phones, 36 used), PER 0.8554 vs null 0.8507, delta **+0.0047**, rate 3.31/s;
   ep10 usage 4.995 / 4.952 bits (39 used), PER 0.8968 vs null 0.9067, delta **-0.0099**, rate
   9.10/s. Both reproduced the banked `per.json` (the reader asserts it) and ran in 9-21 s.
3. Existing suite after the config edit -- `python -m speech_llm.sae.emc.test_infomax` from the
   setup dir (needs `sis_env/bin` on PATH for `black`): **exit 0, all 11 tests pass**, including
   `test_the_arm_configs_differ_from_ctrl_only_in_the_infomax_arguments`, which re-asserts
   `PackedBlankfreeTrainJob.YtvrSez8z9Wf` (node_d) and `...ks7CbtlvpcIL` (node_a, running, unmoved).
4. Load-only census -- `sis --config config/sae_4a_infomax_pack.py console --script -c ...` (sis_env
   python), exit 0, no traceback: **JOBS 315 (was 265: +50, exactly the new readers), TARGETS 536
   (was 436: +100)**, exactly two `PackedBlankfreeTrainJob`s, `YtvrSez8z9Wf` and `ks7CbtlvpcIL`,
   both UNMOVED; 25 `PosteriorEntropyJob` + 25 `SymbolUsageNullJob`, all distinct hashes, all 50
   aliases under `sae/4a/infomax_pack/<arm>/ep<k>/reads/`.

Loading is not a result: none of the 50 jobs has run. The manager on
`config/sae_4a_infomax_pack.py` (pid 1230487, started 17:52) was alive and past graph loading before
the config edit, as the brief required; a live manager holds the graph it loaded, so these reads
enter the graph at the next manager restart, which is the orchestrator's call. No manager was
started, stopped or restarted here.

## 5. Left undetermined / for the phase file

**Which inventory the 3-bit low-inventory clause is read in.** Amendment 5 says "bits over the 40
outputs", but the numbers it is calibrated against (rate arms 4.3-4.5 bits, gold 4.8, the S3b-C arms
collapsed onto 3-4 symbols; `SAE_4A.md:984`) were computed by `emc_hyp_inspect` on the SIL-REMOVED
decode over the 39 phones. The two differ by the SIL share, and on real data that is not cosmetic:
`ctrl_50` ep1 reads **2.824 bits over the 40 outputs but 3.031 over the 39 phones** -- opposite
sides of the 3-bit threshold. The job therefore banks BOTH, named: `symbol_entropy_bits` (=
`symbol_entropy_bits_40_outputs`, the brief's and amendment 5's wording) and
`symbol_entropy_bits_39_phones` (the convention of the banked comparison numbers). The assumption
made in code is only that both are reported; **which one the gate reads is a decision for the phase
file, not for the implementer**, and until it is pinned the ep10 read should quote both.

Two further notes, no action taken: (a) both readers are registered on dev-other only, as the brief
and the phase's gate state; dev-clean would double the 50 jobs if ever wanted. (b) the entropy
reader's pooled per-output-frame mean is the primary number (the monitor convention); the
per-utterance mean and sd are reported beside it because the pooled mean weighs long utterances
more (ctrl_50 ep1: pooled 3.0412 vs utterance-mean 2.9462).

## 6. Commits (explicit paths, nothing pushed)

* `recipe/2025-10-speech-llm` (branch `haotian_modality_matching_jupiter`): the new job module, the
  new test module and the config edit. Other people's uncommitted files in that checkout
  (`config_sae_1g_v1.py`, `config_sae_3e1_d6_swap_cont_v1.py`) were NOT staged.
* `recipe/i6_experiments` (`.../exp_logs/SAE/reports/impl_infomax_reads_2026-09-20.md`): this report.
