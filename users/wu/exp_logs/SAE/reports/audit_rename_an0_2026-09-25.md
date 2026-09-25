# Audit: SAE_4A_rename AN-0 under Amendment R1 (RenameEmStepJob.sXyLgUqfPBPv), 2026-09-25

VERDICT: CONFIRMED_WITH_CORRECTIONS. Under the registered rule the outcome is AN-0 DEAD. I re-derived every
number the verdict uses from the job's saved keys, and my own unit weights reproduce the job's values exactly.
The step is not a no-op, the LM prior is not permuted along with phi, the derangement was applied, and the
verdict logic matches the registered text. The correction concerns what DEAD licenses. It licenses not funding
TP-A1 on AN-0 and dropping AN-3's grid, as registered. It does not license "an LM-led E-step cannot rename".
AN-0 measures one step, from a maximally sharp key channel, with 41 % of train frames deranged, on one seed.
TP-A1's own premise is the opposite regime: a flat channel at the start, iterated.

Inputs read: SAE_4A_rename.md (Shared inputs l.89-95, Amendment R1 l.97-116, AN-0 l.118-125, TP-A1 l.206-227);
job dir /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/rename_emstep_jobs/RenameEmStepJob.sXyLgUqfPBPv
(output/rename_emstep.json, output/report.txt, log.run.1, info); rename_emstep_jobs.py and test_rename_emstep.py
at d2bbd4c7 (the file is unchanged since that commit, which was made at 01:25, and the run started at 01:40;
the dependencies blankfree_emtable, phi_relabel_s, key_search_jobs, unit_key, s1a_job, blankfree_genmarg_jobs
and prior were last committed on or before 09-24 and are clean); lattice.py (_prior_term, PriorHistory);
reports impl_rename_an0_r1, review_rename_an0_launch, review_rename_an0_r1 and extract_rename_an0 (all dated 2026-09-25).

## 1. Re-derived numbers (independent code; no project module imported)

Scripts (session scratchpad): audit_an0.py, audit_an0_margin.py and audit_an0_lmflip.py under
/tmp/claude-34349/-e-project1-spell-wu24-2026-07-13-unsupervised/a2d26b34-6240-4bb2-b34b-ab87c54fe915/scratchpad/.

- Weights: I bincounted the four units.train.shard*.hdf over the 28,254 train.segments tags. The total is
  15,275,716, which equals the job's weights_total and PhiFromKeyInitJob.f0jaGuiJVe6A key_table.npz "counts"
  element by element.
- Gold key: from GoldUnitKeyJob.sLnMRRd2qO0t key.json. It equals key_table.npz "key".
- I recomputed identity, many-to-one and restore from each cell's saved key_before and key_after, with those
  weights. restore is sum w[(k0 != gold) & (k1 == gold)] / sum w. All 16 cells match the job to a maximum
  absolute difference of 0.0. For the read row (gold_key__5pair_s1):

| cell | restore | identity before -> after | gold-row identity after (guard) | VALID |
|---|---|---|---|---|
| plain/1 (= rate_neutral/1) | 0.0000 | 0.5931 -> 0.5916 | 0.9964 | yes |
| plain/2 | 0.0019 | 0.5931 -> 0.5934 | 0.9925 | yes |
| plain/4.4 | 0.0001 | 0.5931 -> 0.5771 | 0.9888 | yes |
| rate_neutral/2 | 0.0032 | 0.5931 -> 0.5963 | 0.9950 | yes |
| rate_neutral/4.4 | 0.0053 | 0.5931 -> 0.5985 | 0.9834 | yes |
| plain/10 (report only) | 0.0000 | 0.5931 -> 0.2996 | 0.5817 | no |
| rate_neutral/10 (report only) | 0.0209 | 0.5932 -> 0.5629 | 0.8982 | no |

- I also recomputed key_before independently as argmax_s m0(u|s) pi(s), with m0 the key_table rows permuted
  by g and pi the cell's saved pi. It equals the job's key_before on 500 of 500 units in every cell of both rows.
- Verdict re-applied: no cell is OPEN. The DEAD set is plain/2, plain/4.4, rate_neutral/2 and rate_neutral/4.4,
  all VALID, with a maximum restore of 0.0053 against the 0.05 bar. That is DEAD. The extraction's figures
  (restore 0 to 0.0053 on VALID cells, every gold row VALID at lambda <= 4.4, rate_neutral/10 at 0.021 and not
  valid) are correct.

## 2. Does the step do anything? Yes

I split the key transitions by train-weight share, recomputed from the saved keys:

| row / cell | wrong->right | right->wrong | wrong->other wrong | wrong->same wrong | units re-keyed (weight) |
|---|---|---|---|---|---|
| deranged plain/1 | 0 | 0.0016 | 0.0121 | 0.3947 | 8 (0.014) |
| deranged plain/2 | 0.0019 | 0.0016 | 0.0195 | 0.3855 | 13 (0.023) |
| deranged plain/4.4 | 0.0001 | 0.0161 | 0.0331 | 0.3738 | 35 (0.049) |
| deranged rn/2 | 0.0032 | 0 | 0.0139 | 0.3898 | 9 (0.017) |
| deranged rn/4.4 | 0.0053 | 0 | 0.0214 | 0.3802 | 15 (0.027) |
| deranged rn/10 | 0.0209 | 0.0512 | 0.1882 | 0.1977 | 115 (0.260) |
| gold plain/1, 2, 4.4 | 0 | 0.0036, 0.0075, 0.0112 | 0 | 0 | 3, 5, 10 |
| gold rn/2, 4.4, 10 | 0 | 0.0050, 0.0166, 0.1018 | 0 | 0 | 4, 8, 51 |

- The E-step posterior moves, but not far enough to flip keys. These are the job's own "moves", the shares of
  the gold-p units' expected frames. At rn/4.4, AO carrying N sends 0.106 of the N-unit frames to N (it was
  0.036 at lambda 1); the share left on AO falls from 0.547 to 0.243, and 0.651 goes elsewhere. T-unit frames
  sent to T rise from 0.055 to 0.108. The mean share sent to the correct name over the 10 moved symbols is
  0.019, 0.031, 0.045 and 0.037 at rate-neutral lambda 1, 2, 4.4 and 10.
- In every VALID cell, and for every moved symbol, the frames sent to the correct name stay below the frames
  that stay on the wrong name. So no key flips: key_after is about argmax_s N(s,u).
- The largest per-symbol key restoration is AO carrying N at rn/4.4: 0.051 of the N-unit weight.
- The M-step table changes substantially. S_1 falls by 0.31 on the gold row and by 0.39 on the deranged row at
  lambda 1, with the same smoothing on both sides. N(s,u) and m1 are not saved, so I cannot give a direct table
  distance.
- Nine units (weight 0.0038) do not occur in the 300 utterances. They re-key to the largest-N(s) symbol, which
  is SIL. They account for all of the gold row's drop at lambda 1 (3 units, 0.0036). They cannot add to restore,
  because SIL is never moved.

## 3. Is the LM prior permuted with phi? No

- relabel_state indexes only the phi state's per-type rows (emb_type.weight, dur_logits).
- The lattice table is the bed's prior_log_bi. It is asserted equal to the banked log_tri, and genmarg._setup
  re-reads it at every batch.
- The trigram history layout is h = p_-2 * 41 + p_-1 in both prior.py and lattice.PriorHistory.
- Empirical check: if the prior were permuted with phi, the deranged row would be an exact relabelling of the
  gold row. The two rows would then have the same log Z, the same tokens per frame and the same S_1. They
  differ:

| quantity | gold row | deranged row |
|---|---|---|
| log Z per frame, lambda 1 | -4.542 | -4.871 |
| log Z per frame, plain/4.4 | -6.201 | -6.473 |
| tokens per frame, plain/4.4 | 0.1618 | 0.1367 |
| S_1 before | 4.563 | 4.886 |

## 4. Derangement, forms, H_LM, utterances

- Pool: 37 non-SIL symbols that hold units in the gold key. OY and ZH are outside it. It equals the job's pool.
- Pairs: RandomState(1).choice(37, 10, replace=False), taken in consecutive pairs, gives AH<->T, AO<->N, P<->S,
  M<->Y and K<->OW, the same set as the job.
- Moved gold frame share: 0.40684. The deranged row's identity before is 0.5931, and gold minus deranged
  identity equals the moved share in every cell. So the derangement was applied as drawn.
- Durations: the non-SIL rows of dur_logits in the gold-key phi are identical (maximum difference 0.0). So
  relabelling the durations is a no-op for non-SIL symbols, and durations carry no name signal.
- Forms: prior_table computes lambda * log P for plain, and adds (lambda - 1) * H_LM for rate_neutral, at
  prior_weight 1.
- The lattice computes scale * prior_weight * table. So the plain form equals prior_weight = lambda bitwise at
  tau = 1. The real test and both reviewers confirmed this on a real utterance.
- H_LM re-derived from prior.npz tri_counts and log_tri is 2.2571083839548 nats (81,559,944 tokens), equal to
  the job's value. Under the LM's own text, rate_neutral keeps the mean per-token score at -2.257 for every
  lambda.
- The rate-neutral form is not rate-neutral on the acoustics: the gold row's tokens per frame are 0.2356,
  0.2357, 0.2524 and 0.3252 at lambda 1, 2, 4.4 and 10. This is disclosed, as registered, and absorbed by the
  guard.
- Utterances: re-deriving s1a_job's selection (sorted tags, RandomState(0) permutation, first 300) gives the
  job's first 5 tags and 164,906 frames. Every cell has 300 utterances and no Z = 0 utterance.
- Constants trace to the registration or to A11/A12: 300 utterances, 5-pair derangement with seed 1, lambda
  grid, 0.9/0.1 smoothing, tau 1, pseudo-count 1e-3, durinit, guard 0.95, bar 0.05.

## 5. What the reading licenses, and why the consequence clause overreaches

These are frame findings. They do not change the outcome.

1. **One step, argmax statistic.** restore counts units whose argmax_s N(s,u) moves to the right name in a
   single E-step. The E-step does move mass toward the right names as lambda rises (section 2). Whether that
   mass compounds over iterations is untested. After one M-step, the right symbol's row holds up to 11 % (0.4-11 % by symbol) of the
   swapped units' frames instead of the floor, so the channel margin at the next E-step is far smaller. TP-A1
   anneals over 12 iterations.
2. **Sharpest possible channel.** The E-step emits 0.81 ML + 0.19/500. Off-key is 3.8e-4, and the margin
   against the swap partner is about 4.4 nats per frame on average (audit_an0_margin.py). TP-A1's rationale
   ("cipher EM succeeds when the channel starts flat and the LM leads") concerns a flat channel at the start,
   which AN-0 does not exercise.
3. **Not a "mostly right context".**
   - Seed 1 swaps AH, T, N and S among others, so 41 % of train frames carry wrong names.
   - In the LM text, 58 % of the moved-phone tokens have a moved immediate neighbour (audit_an0_lmflip.py,
     30,000 lines).
   - The reading rests on one seed. The 1-pair rows and seeds 2-3 exist only in AN-3.
4. **Unexplained magnitude.** This is an audit-side estimate, not a registered statistic.
   - LM gain from flipping a single swapped token back, with its neighbours as deranged, at lambda 1: median
     6.6 nats over all moved tokens; 15.9 for true N and 19.2 for true AO.
   - Multiplied by 4.4, this exceeds a rough per-segment channel margin (about 4.4 nats per frame over a few
     frames).
   - Yet at rn/4.4 only 0.4-11 % of the swapped units' mass reaches the right name, while 24-65 % goes
     "elsewhere".
   - Where it goes, in part: SIL share rises from 0.079 to 0.118. The uniform OY and ZH rows (emission 0.002
     per frame, 1.66 nats per frame above off-key) take 5.7 % of frames at plain/4.4, against 0.45 % at
     lambda 1.
   - So the lattice seems to resolve the LM pressure by re-segmenting and relabelling rather than by renaming.
     The code checks (sections 3 and 4) found no defect.
   - These files cannot tell whether this is lattice behaviour or an undetected E-step issue. Settling it
     would need the per-cell N(s,u) (not saved), or the MAP paths on a few utterances, for the gold and
     deranged rows at rn/4.4.

Licensed: "one LM-weighted E-step at lambda <= 4.4 (plain or rate-neutral) does not flip the keys of a
material share of swapped units from the deranged gold-key phi (restore at most 0.0053 against 0.05), so TP-A1
is not funded on AN-0 and AN-3's grid is dropped, as registered".

Not licensed:
- "no LM-weighted E-step moves a whole symbol" beyond this one-step, sharp-channel, seed-1 operating point;
- "TP-A1 could not work";
- any statement about iterated EM, a flat-channel start, 1-pair contexts or other seeds.

The registered consequence text ("TP-A1's mechanism is dead") states more than the measurement shows. A failed
gate licenses "not funding it", not "it would not have worked".
