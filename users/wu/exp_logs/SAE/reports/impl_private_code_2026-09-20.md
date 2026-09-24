# impl: private-code analysis of the budget control (2026-09-20)

Brief: one registered analysis read of what the content-free blank-free recognizer's output
encodes, on the budget control's existing kept checkpoints, consuming banked artifacts as FROZEN
paths (`SAE_4A_infomax.md` "Design" amendment after code review; `SAE_4A_objective.md` section 6).
Part E (label-free decipherment) was added mid-task by the coordinator and is included.

STATUS: DONE. Registered, tested, census-clean, and smoke-run end to end on ctrl_50 ep10 dev-other
(the numbers below are that local run of the job's own `run()`, NOT a banked sisyphus result --
no manager was started).

**Amended 2026-09-20 (second round):** part A gained the one-to-one-WITH-DROP relabelling and part
E gained E5, the hard relabelling against a null.  **Amended again (third round, after
`audit_private_code_2026-09-20.md` items 2-4, still before anything is launched):** the
many-to-one and with-drop rows are now RE-COLLAPSED, E5's null is a matched many-to-one null with
the identity score beside it and a stricter rule, and the E table carries the gold reference with
SIL-free rows.  Every number below is from a re-run after those fixes.

## What exists on disk (the inputs question)

| input | present? | pin |
|---|---|---|
| ctrl_50 greedy decode, ep1 / ep4 / ep10, dev-other AND dev-clean | yes | `BlankfreeGreedyPerJob.{XN6vhGGyKQu5, VhipueEiV9cf, Vz6QOYliPU40, gAmloZ1hqFfq, dG4n46xTRSl0, GYhOaiWJF2Yv}` (`greedy_raw.json`, `greedy_phones.json`, `per.json`) |
| PER-FRAME log posteriors of those same decodes | yes | the `ReturnnForwardJobV2` each PER job consumes (`{FI29gVlt2cCT, 9dEcv183aUBo, AY6gwxHxHcUR, 69Oir6kenfkh, QMYeLWX1G7Na, Mx6zqBJhIxwh}` / `posteriors.hdf`, `[frames, 40]` log posteriors at the 60 ms clock) -- so NO new GPU forward was needed |
| ctrl_50 ep25 decode | NO | `alias/sae/4a/blankfree/budget_pack/ctrl_50/ep25/*/per` is a dangling link; ep25 is not registered (see "when ep25 lands") |
| K = 500 reverse UNITS for dev | YES, both dev splits | `BlankfreeVadHdfJob.SAjz8y1cT06g/output/units.dev-{other,clean}.shard0.hdf` (781,130 / 831,372 retained 50 Hz frames), the same VAD job every ctrl_50 decode reads. No train-side forward is needed and part B runs on dev |
| MFA gold FRAME alignments | yes, but NOT as a sisyphus job | there is no banked MFA-alignment job in the graph; the campaign's frame gold is the gilkeyio parquet cache `/e/project1/spell/common_hf_home/hub/datasets--gilkeyio--librispeech-alignments/...` rasterised by `repr_audit.frame_phone_labels` (`analysis/emc_target_vs_gold.load_gold_frames`, `init_jobs.SeedGoldPhonesJob`). It covers ALL dev utterances, not only the S1a 500: the ep10 run joined 2864 / 2864 dev-other utterances, 0 dropped, 0 missing |
| gold phone strings (PER reference) | yes | `s0b.GOLD_PHONES` = `GoldPhonesJob.ZGSp0hxyd2YP` |
| the arm's own text prior | yes | `PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz`, the unbiased priorshuf refit the packed `node_a` training job itself consumes (its `info`: `INPUT: .../PhoneNgramPriorJob.RtzbESkOedsT/output/prior.npz`) |
| speaker ids | yes | `s1a_job.speaker_of` on the utterance tag (33 speakers on dev-other) |

## Files

* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/private_code.py` (NEW module -- a new file, so
  no file-sha `__sis_version__` anywhere in the campaign moves): the primitives (`align_pairs`,
  `hungarian_labelling`, `majority_labelling`, `nmi`, `conditional_entropy_bits`, `kl_bits`,
  `bigram_joint`, `run_lengths`, `load_gold_frames`, `em_decipher`, `viterbi`, `forward_loglik`,
  `prior_score_per_token`) and two CPU in-process jobs, `PrivateCodeAnalysisJob` (parts A-D) and
  `SymbolDeciphermentJob` (part E).  Every convention is pre-registered in the module/class
  docstrings, including "gold is used for the READ only" and "THE FIT USES NO LABELS ... PER is
  therefore computed only after the fit".
* `recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_private_code.py` (NEW): the tests below.
* `.../librispeech/configs/config_sae_4a_private_code_v1.py` (NEW): every input as a `tk.Path` with
  a `hash_overwrite`; aliases `sae/4a/private_code/ctrl_50/ep<k>/<split>` (+ `/decipher`), outputs
  `.../sae_4a_private_code/ctrl_50/ep<k>/<split>/{private_code,decipher}.{json,md}`.
* `config/sae_4a_private_code.py` (workspace, 2 lines, untracked like the rest of `config/`).

No existing file was edited, so no banked hash can move.

## Jobs registered (load-only census, 8 jobs, nothing else)

`sis --config config/sae_4a_private_code.py console --script -c "<print sis_graph.jobs()>"`:

```
PrivateCodeAnalysisJob.{DMAmCpn0L5mB, VAYV87DbdDwU, eLdlRYU1YtQZ, ssh9wJM8pxjI, x79av2mvTxo3, xuCBRVe3CzMR}   ep{1,4,10} x dev-{other,clean}
SymbolDeciphermentJob.{UrqIAKvjW0QK, qV0ZJNWBl3z4}                                                            ep{4,10} dev-other
```

No training job, no forward job, no producing job is in this graph.  Measured cost of the local
smoke run: parts A-D 31 s, part E 508 s per read (rqmt 4 CPU / 32 GB / 4 h and 4 / 16 / 4).

## Conventions worth knowing before reading a number

* SYMBOL = one of the 40 outputs, named by `prior.PHONES` (nominal names only).  OUTPUT FRAME = 3
  VAD-retained 50 Hz frames = 60 ms (stride 3), assigned retained frames `3f, 3f+1, 3f+2`.
* LABELLING = symbol -> phone or DELETE.  The IDENTITY labelling (phone symbol -> its own name,
  SIL -> DELETE) IS the banked scoring convention, so every relabelled PER lives in the same
  space.  The confusion matrix comes from the identity alignment; a relabelled PER is recomputed
  with a FRESH alignment.
* The one-to-one (Hungarian) PER can come out ABOVE the identity PER: the Hungarian maximises
  matched pairs and is blind to the DELETE slot, and the symbol it sends to DELETE is in general
  not SIL, so the SIL tokens stay in the hypothesis and score as insertions.  This is stated in
  the rendered report next to the row.
* At the FRAME level SIL is a real gold label, so the one-to-one labelling there is a genuine
  40 x 40 permutation and the reported quantity is the frame error rate (the PER analogue; an edit
  distance between two equal-length frame strings is not one).
* Entropies / MI / KL in BITS; NMI = I / sqrt(H H), with I, H(X), H(Y) beside it.
* Part A item (i) asserts the recomputed PER reproduces the banked `per.json` to 0.005 (it
  reproduces to 1e-12 at ep10), AND that the collapsed argmax of `posteriors.hdf` equals
  `greedy_raw.json` token for token for every utterance -- the frame reads are the same decode.

## The full table, ctrl_50 ep10 dev-other (local run of the job, 2864 utterances)

### A. phone identifiability

| read | value |
|---|---|
| PER as scored (identity labelling) | 0.896793 |
| PER banked (`per.json`) | 0.896793 |
| PER after best one-to-one relabelling (Hungarian) | 0.918748 |
| PER after best one-to-one-with-drop relabelling (re-collapsed) | 0.840288 (12 of 40 symbols dropped) |
| PER after best many-to-one relabelling (majority, re-collapsed) | 0.855191 |
| NMI(symbol, phone), aligned token pairs | 0.0562 |
| I / H(symbol) / H(phone), bits | 0.2760 / 4.9923 / 4.8271 |
| symbols with aligned mass (of 40) | 40 |

The plain one-to-one exceeds the identity PER because the square assignment has one DELETE slot
and spends it on UH, so SIL's 7639 tokens (1513 of them insertions) stay in the hypothesis.  The
with-drop variant gives every symbol a drop option priced with the insertions the drop removes; it
drops 12 symbols including SIL, insertions fall 13013 -> 3070 and the PER falls to 0.8403, below
the identity read.  That is the one-to-one number to quote; the old column is kept.

Re-collapse convention (audit item 2): a many-to-one map, and a drop, can make two neighbouring
labels equal, and the recognizer would emit ONE token there, so those two rows are now re-collapsed
before scoring, the convention part E always used.  The identity row must NOT be (it has to
reproduce `per.json`, whose string drops SIL without re-collapsing) and the plain one-to-one row
keeps that same banked convention, so it stays the deletion-blind reference the with-drop row is
read against; `recollapsed` in each json row says which.  Both re-collapsed numbers reproduce the
audit's independent recomputation exactly (0.855191 and 0.840288).

Frame level (MFA on 2864 / 2864 utterances, 261,295 output frames):

| read | value |
|---|---|
| frame error, identity labelling | 0.919137 |
| frame error, best one-to-one | 0.765139 |
| frame error, best many-to-one | 0.698747 |
| NMI(symbol, phone), frames | 0.2559 |
| I / H(symbol) / H(phone), bits | 1.2697 / 4.9853 / 4.9403 |

### B. unit code (K = 500, unit-frame level)

| read | value |
|---|---|
| H(symbol \| unit), bits | 2.5027 |
| H(unit \| symbol), bits | 6.2206 |
| NMI(symbol, unit) | 0.3770 |
| H(symbol) / H(unit), bits | 4.9870 / 8.7049 |
| symbol mass on the top-1 unit -> symbol map | 0.4953 |
| REFERENCE H(gold phone \| unit) / H(unit \| gold phone), bits | 1.9197 / 5.6808 |
| REFERENCE NMI(gold phone, unit) | 0.4610 |
| REFERENCE gold mass on the top-1 unit -> phone map | 0.6143 |

### C. nuisance

| read | value |
|---|---|
| NMI(symbol, speaker), 33 speakers | 0.0065 |
| NMI(gold phone, speaker) | 0.0035 |
| symbol-usage entropy, tokens / frames, bits | 4.9945 / 4.9853 |
| frame mass in the top-5 symbols | 0.2790 (AH, IH, S, SIL, L) |
| mean symbol run length | 1.492 output frames (89.5 ms) |
| mean MFA gold phone duration | 4.198 raw frames (84.0 ms), median 4.0 |

Per-symbol (10 symbols with the most runs), mean run in ms vs the gold mean duration of the phone
of the same NAME: AH 95.0 / 64.2, IH 92.4 / 65.6, S 86.4 / 103.7, L 88.2 / 79.0, SIL 95.4 / 116.4,
N 77.2 / 68.2, T 84.8 / 73.8, R 82.8 / 68.7, D 89.3 / 65.3, UW 86.7 / 93.2.

### D. sequence statistics

| read | value |
|---|---|
| KL(symbol bigram \|\| text bigram), bits | 2.0076 |
| KL(text bigram \|\| symbol bigram), bits | 3.1369 |
| KL(symbol bigram \|\| gold bigram), bits | 1.3342 |
| KL(gold bigram \|\| symbol bigram), bits | 1.0734 |
| REFERENCE KL(gold bigram \|\| text bigram), bits | 0.5554 |
| REFERENCE KL(text bigram \|\| gold bigram), bits | 2.4485 |
| symbol runs of exactly 1 output frame | 0.5956 (of 175,149 runs) |

### E. label-free decipherment (same checkpoint, dev-other; fit on the even half, no labels)

EM kept restart 0 after 50 iterations (the iteration cap, not the 1e-4 tolerance),
log likelihood per token -3.011050.

| read | fit half (1432) | held half (1432) |
|---|---|---|
| PER, identity labelling (as scored) | 0.894902 | 0.898720 |
| PER, DECIPHERED (label-free) | 0.851787 | 0.854799 |
| PER, Hungarian one-to-one (label-using) | 0.916386 | 0.921153 |
| trigram log p per token, identity | -3.9783 | -3.9949 |
| trigram log p per token, deciphered | -2.1215 | -2.1239 |
| trigram log p per token, Hungarian | -4.9251 | -4.9372 |
| HMM log lik per token under the fitted cipher | -3.0108 | -3.0143 |

| read | value |
|---|---|
| symbols where the deciphered map agrees with the Hungarian map | 11 of 40 |
| token mass on those symbols | 0.3960 |
| mean H(phone \| symbol), bits (token-mass weighted) | 1.7982 |
| mean H(symbol \| phone), bits (emission rows) | 1.7405 |

The fit/held split of every E row is within 0.004 PER and 0.02 nats, i.e. the cipher generalises
across halves; the deciphered labelling buys 0.043 PER over identity and 2.0 nats per token of
trigram score, while the deciphered and the label-using Hungarian maps agree on only 11 symbols.
Reading these numbers against the three hypotheses is the orchestrator's call, not mine.

### E5. hard relabelling (no search) against a MATCHED null, with the gold reference

E2's deciphered string is the Viterbi path, prior-optimised by construction, so -2.12 vs -3.98
carries no evidence.  E5 applies the fitted cipher's argmax phone per symbol (deterministic,
many-to-one, the map E3 scores) to the raw collapsed greedy string, re-collapses, and scores it
against two nulls of 20 draws each (seed 0, same draws on both halves): the MATCHED null, the
fitted map's own target vector permuted over the 40 symbols (same image multiset, same collapsing
power -- the null the rule uses), and the BIJECTION null of the second round, kept and labelled as
the weaker one.

| read | fit half | held half |
|---|---|---|
| trigram log p per token, identity labelling | -3.9783 | -3.9949 |
| trigram log p per token, hard relabelling | -3.8123 | -3.8373 |
| matched many-to-one null (mean / sd / max) | -6.9705 / 0.4916 / -6.1985 | -6.9738 / 0.4890 / -6.2467 |
| bijection (permutation) null (mean / sd / max) | -8.4257 / 0.3918 / -7.8346 | -8.4245 / 0.3908 / -7.8242 |
| margin over the matched-null max / over identity | 2.3862 / 0.1659 | 2.4093 / 0.1576 |
| **PRE-REGISTERED reading (both margins > the matched sd)** | **False** | **False** |
| PER, hard relabelling | 0.851496 | 0.853045 |
| PER, matched null (mean / sd / max) | 0.868027 / 0.013325 / 0.898546 | 0.870607 / 0.013510 / 0.901829 |
| PER, bijection null (mean / sd / max) | 0.941122 / 0.008011 / 0.954225 | 0.944596 / 0.008118 / 0.957753 |

The rule now has two parts, banked separately: `exceeds_matched_null_max_by_sd` is **True** on both
halves (margin 2.39 / 2.41 against sd 0.49), `exceeds_identity_by_sd` is **False** (margin 0.17 /
0.16).  So at ep10 the hard relabelling beats a shape-matched null comfortably and beats doing
nothing by far less than the null's own spread; the conjunction, which is what the docstring
pre-registers, does not read.  About 1.5 nats of the second round's 4.6-nat margin was bought by
the many-to-one shape alone (matched mean -6.97 vs bijection mean -8.43), as the audit found.

Like-for-like with gold (gold is SIL-free, so these rows drop the SIL tokens; reference only, no
rule reads them):

| read | fit half | held half |
|---|---|---|
| trigram log p per token, GOLD transcripts | -3.1898 | -3.2087 |
| trigram log p per token, identity, SIL-free | -4.6174 | -4.6402 |
| trigram log p per token, hard relabelling, SIL-free | -4.5685 | -4.5979 |

Gold and the SIL-free identity row reproduce the audit's numbers exactly (-3.1898 / -3.2087 and
-4.6174 / -4.6402; whole split -3.1992).  My SIL-free HARD row is -4.5685 / -4.5979 against the
audit's -4.5447 / -4.5746: the 0.024-nat difference is the order of the two operations.  I
re-collapse the relabelled string and then drop SIL without re-collapsing, which is the banked
`greedy_phones` convention and is why the identity row matches; dropping SIL first (or collapsing
again afterwards) merges neighbours that SIL had separated.

## Checks run

* `pytest recipe/2025-10-speech-llm/src/speech_llm/sae/emc/test_private_code.py` -- 11 passed, 42 s
  (8 of the first round plus three fixtures):
  part A on a 3-utterance fixture under a known permutation (alignment counts equal
  `eval_jobs.edit_counts`; the Hungarian AND the majority map recover the permutation on every
  occurring symbol; relabelled PER exactly 0; identity PER > 0.5); C's NMI on fixtures (perfect
  dependence 1, independence 0, a hand 2 x 2 with every entropy checked); B's conditional
  entropies on a hand joint table; `run_lengths` and KL; part E on a fixture (phone strings
  sampled from a fitted bigram prior, passed through a random permutation of the 40 symbols):
  40 / 40 symbols recovered by the label-free EM in 18 iterations and deciphered PER 0.0
  (bars: >= 35 and < 0.05);
  a deterministic-cipher Viterbi check (bigram, trigram and a length-1 string) and the prior-score
  convention against `PhoneNgramPrior.log_prob`; and that the no-labels rule is in the job
  docstring.  NEW: a 4-symbol / 3-phone fixture in which the plain Hungarian must keep an
  insertion-heavy SIL analogue (4 matched pairs beat the rival's 2) and drop a useful symbol, while
  the with-drop assignment reverses both and scores PER 4/38 against the plain variant's 32/38; and
  E5 on the cipher fixture, where the label-free hard map beats all 20 null permutations on the
  prior score and on the PER (< 0.05, below the null min).  THIRD ROUND: the E5 fixture now builds
  the MATCHED null (asserting each draw has the fitted map's image multiset) and checks the new
  two-part rule; plus a hand fixture for the re-collapse convention (a map sending two neighbours
  to one phone: 1 deletion collapsed against 2 insertions uncollapsed) and that the job docstring
  pre-registers which rows are re-collapsed.
* **Fourth round (`version` parameter).**  The eight jobs had in fact been LAUNCHED and finished
  under the second-round code, and the third round moved no hash, so the corrected reads would
  never have run.  Both classes now take a hash-relevant `version` (default 2, banked in the json);
  the census shows eight NEW hashes and none of the old eight, and the eight finished dirs are left
  untouched:
  `PrivateCodeAnalysisJob` ep1 dev-other `j5ybPkSFOSBq`, ep1 dev-clean `1bNrZdet9JcH`, ep4 dev-other
  `UPuAcKSAI5oK`, ep4 dev-clean `QY8blUARrLxQ`, ep10 dev-other `cQbcIJtOamLm`, ep10 dev-clean
  `OIDSbcHXTzsP`; `SymbolDeciphermentJob` ep4 `GpiTxaoRCZXG`, ep10 `m8EsFhL6ysqu`.  The old dirs
  (`DMAmCpn0L5mB`, `VAYV87DbdDwU`, `eLdlRYU1YtQZ`, `ssh9wJM8pxjI`, `x79av2mvTxo3`, `xuCBRVe3CzMR`,
  `36NfY7XDOL3f`, `FkH0wprbfjaN`) hold the pre-audit numbers and must not be read.
* Load-only census after the second round (`sis ... console --script`): 8 jobs, exactly the 6 + 2
  registered here; the two `SymbolDeciphermentJob` hashes MOVED with the new `null_permutations` /
  `null_seed` parameters (`36NfY7XDOL3f` ep4, `FkH0wprbfjaN` ep10), the six `PrivateCodeAnalysisJob`
  hashes did NOT (this module stamps no file sha, so a body-only change is hash-neutral).  The
  third round changes only job bodies, so NO hash moves again -- and that is the trap: nothing has
  been run (`work/speech_llm/sae/emc/private_code/` is empty), so no stale output survives, but had
  any of the eight already run, its hash would not have forced the re-run of the corrected read.
* Cross-check against the audit's independent recomputation (its own code, from `greedy_raw.json`):
  many-to-one 0.855191 and with-drop 0.840288 reproduce exactly; gold -3.1898 / -3.2087 and the
  SIL-free identity -4.6174 / -4.6402 reproduce exactly; the SIL-free hard row differs by 0.024
  nats for the stated convention reason.
* End-to-end local run of both jobs' own `run()` on ctrl_50 ep10 dev-other, re-done after the
  third-round fixes (every number above is from that run; A-D 35 s, E 654 s with both nulls): the
  PER reproduction assert and the per-utterance "collapsed argmax == banked greedy string" assert
  passed for all 2864 utterances, and the MFA join lost no utterance.

## Undetermined by the brief -- what I chose, and why it is a routine choice

* Part E's EM init noise scale (0.1, multiplicative on a uniform table) and the restart seeds
  (0, 1, 2).  The brief fixes "uniform plus a small seeded noise, 3 restarts, 50 iterations or
  1e-4 per-token improvement"; the scale and seed are job parameters with these defaults.
* "Rescore with the trigram afterwards" is implemented as: EM with the prior's BIGRAM transitions,
  then the final Viterbi decode with the prior's TRIGRAM table (same context convention as
  `PhoneNgramPrior`, no EOS term).
* "The text phone bigram" of part D is the arm's own banked prior as `p_uni(a) p_bi(b | a)`; the
  dev GOLD phone bigram is reported beside it as a second reference.  Both KLs use add-one
  smoothing on the empirical table.
* "The emission table's entropy per symbol" is H(phone | symbol) in bits, token-mass weighted
  (the row entropies H(symbol | phone) are reported beside it).
* The deciphered map is `argmax_p p(symbol | p) p_uni(p)` (the MAP phone of a symbol); the full
  emission table is in the json so any other reading can be recomputed.
* C's "NMI(symbol, speaker) via per-utterance symbol histograms" is the joint of (token, speaker of
  its utterance), i.e. the per-utterance histograms summed per speaker; the gold row is built the
  same way.
* **The drop option's price (second round).**  The brief says "cost = that symbol's token count,
  i.e. its tokens are removed from the hypothesis before scoring".  Charged literally -- drop costs
  `n_tok(s)` while keeping at phone `c` costs `n_tok(s) - conf[s, c]` -- the drop is never worth
  taking and the variant is a no-op identical to the existing one-to-one.  I priced it the way the
  recomputed PER prices the removal, which is what the parenthetical describes: removing the
  symbol's tokens turns its `aligned(s)` tokens into deletions and makes its `ins(s)` insertions
  disappear, so drop costs `aligned(s)` and the assignment maximises `conf[s, c]` for a kept symbol
  and `ins(s)` for a dropped one.  This is the one place where I read the wording rather than
  followed it literally; the derivation is in `hungarian_drop_labelling`'s docstring.
* **E5's "argmax phone under the fitted emission table"** is read as the map E3 already reports,
  `argmax_p p(symbol | p) p_uni(p)` (the MAP phone), not the bare `argmax_p p(symbol | p)`, so that
  the E3 agreement number and the E5 rows describe the same mapping.  The full emission table stays
  in the json.
* `spread()` also banks the null's `min` and `n` in the json; the rendered table shows mean / sd /
  max as the brief asks.
* **The third round's rule wording.**  "Exceeds both the matched-null max AND the identity score by
  more than the matched-null sd" is implemented with the sd margin on BOTH comparisons (the literal
  parse, and the strict one).  It decides the verdict at ep10: with the sd margin the reading is
  False on both halves; without it on the identity side it would be True.  Both components are
  banked separately (`exceeds_matched_null_max_by_sd`, `exceeds_identity_by_sd`), so the other
  reading needs no re-run -- but the docstring pre-registers the strict one.
* **The matched null reuses the bijection null's 20 draws** (the fitted map's target vector
  permuted by the same 20 permutations, seed 0), which is exactly "permute the fitted map's phone
  assignments over the 40 symbols, seed 0" and keeps one seed for both rows.
* **The plain one-to-one row is deliberately NOT re-collapsed** (the brief named the many-to-one
  and with-drop rows): it is the banked-convention, deletion-blind reference the with-drop row is
  read against.  Re-collapsing it would move a number already quoted in the phase file; say so if
  that is wanted instead.

## When ep25 lands

Append 25 to `EPOCHS` in the config and add its three job hashes to `PER_JOBS` / `POST_JOBS`
(and it joins `DECIPHER_EPOCHS` if part E is wanted there).  No hash below moves.
