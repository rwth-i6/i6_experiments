# Audit: A18 (b) durinit basin lift read, DURINIT BASIN LIFTS (auditor, 2026-09-25)

Verdict: CONFIRMED_WITH_CORRECTIONS. The verdict DURINIT BASIN LIFTS holds as registered, and every per-arm PER re-derives exactly from the decode outputs. There are four corrections:
- The node-fault exposure check missed the forwards that carry the verdict.
- The paired rows did finish.
- The frame of the brief is wrong about label use.
- Two of the three paired rows sit inside the seed spread.

Nothing was edited, cleared or rerun. My scripts and logs are in the session scratchpad, under `audit_a18b/`: recompute.py, recompute.log, recompute.json, forensic.py, forensic.log and forensic2.log.

## 1. Per-arm PER, recomputed from the decode outputs
Method: I read each BlankfreeGreedyPerJob's `posteriors.hdf`, then took the per-frame argmax, collapsed repeats, dropped SIL, and scored with my own Levenshtein against `GoldPhonesJob.ZGSp0hxyd2YP/output/gold.json` (dev-other). I did not use `eval_jobs.edit_counts` or the reader's JSON. There are 19 decodes: 4 arms x ep1/2/4/8, plus the 3 A17 (i) ep8 baselines.
- Every decode has 2,864 utterances, and its tag set equals the gold set. The reference has 177,275 phones.
- Every posterior is finite and log-normalised (max |logsumexp| 3.2e-7). No hypothesis is empty.
- My hypotheses equal each job's greedy_phones.json, and every PER equals per.json exactly (abs diff 0.0).

| Arm | ep1 | ep2 | ep4 | ep8 | errors at ep8 | class at ep8 |
|---|---|---|---|---|---|---|
| gold_key | 0.2752 | 0.2581 | 0.2351 | 0.2078 | 36,840 | LIFT |
| g_dur | 0.2749 | 0.2371 | 0.2321 | 0.1931 | 34,234 | LIFT |
| r30_dur | 0.2901 | 0.2638 | 0.2283 | 0.2059 | 36,494 | LIFT |
| r70_dur | 0.3546 | 0.3446 | 0.2783 | 0.2275 | 40,335 | LIFT |

Checkpoint chain at ep8, verified from each job's info:
- The PER jobs bbrCBu3GYtJH, V9FnE2YGnGiC, E8bmqcJz0Gat and cLIbtCqISjr2 read the posterior forwards e9LrruCo4AdR, O1zOKML2oEhv, qtDoaElsEWgZ and 9AUpBk28Ak7F.
- Those forwards read the extract jobs blCT1vvTEwJb, qavyqUhNaCP5, XzOgIwthB2Ll and qPsygTQxQq5J (prefix recognizer.).
- The extracts read `ZUZypSQn7qc0/output/a18_{gold_key,g_dur,r30_dur,r70_dur}/models/epoch.008.pt`.

Features come from `BlankfreeVadHdfJob.SAjz8y1cT06g` feats.dev-other. The A17 baselines use the same features and the same gold file.

Note: "D4" is the 500-utterance subset that only the genmarg reads use. The lift PER covers all 2,864 dev-other utterances.

## 2. Inits and the single delta
- Each lift arm's `reverse_checkpoint_path` goes through an ExtractSubmoduleCheckpointJob (prefix reverse.) to the matching keyinit sub-epoch-48 phi:
  - sN5fh2suaStx: `ge1MKcAPmZIV/output/gold_key/models/epoch.048.pt`;
  - MWW5IWkxO9e0: g_dur;
  - 78bwKGXFV8uc: r30_dur;
  - T4UQLoj320q4: r70_dur.
- The keyinit inits are:
  - gold_key = `PhiFromKeyInitJob.f0jaGuiJVe6A`, which reads `GoldUnitKeyJob.sLnMRRd2qO0t` (each unit's majority label on the train-side MFA alignment) plus durinit;
  - g_dur, r30_dur and r70_dur = `PhiDurinitSwapJob` applied to the supervised reverse fits 16v7R6ztSq1u (gold), HpmOCSCklRsB (rho 0.3, `CorruptSeedGoldJob.xsTnpYUCPdlV`) and OLIUO3BGy0ug (rho 0.7, mnzDC7XJX00r).
- Config diffs of the written returnn.config:
  - between the four arms, only `reverse_checkpoint_path` and the model dir differ;
  - a18_g_dur against A17 (i) a17_gold_em (T18RrTNTdg65) and against A14 (i) rt_em_durfrz_s1 (e9xZa5ElF16P) gives the same two lines.
  - So the arms are in A14 (i)'s form and differ only in phi.
- Both packs report the same RETURNN version (1.20260518...00171dfe.dirty). Between the two pack starts, the speech_llm repo has only config-only commits (b9f676f5, 2d51f35c). Uncommitted working-tree state was not checked.
- The pack trained on jpbo-071-09 and completed ep8 in all arms. There is no "nan" in any arm log.

## 3. Bands and verdict logic
- A4 (phase file, A4): LIFT < 0.50 at ep8, PARTIAL < 0.8164, else NO LIFT. The code (`blankfree_a17_jobs.LIFT_PER`/`PARTIAL_PER`, `lift_class`) uses the same thresholds with strict inequalities.
- `DurinitBasinLiftReadJob.verdict` gives:
  - CANNOT_TELL if any durinit arm has no class;
  - LIFTS if g_dur or r30_dur is LIFT or PARTIAL;
  - DOES NOT LIFT if none of the three lifts;
  - otherwise MIXED.
- The gold-key control enters neither clause. This matches A18 (b) and the build choice.
- Here g_dur (0.193) and r30_dur (0.206) sit about 0.29-0.31 below the LIFT bar, so the verdict does not depend on the reading or the bands.
- Hold rule (State NEXT 1, review section 6): clause (1) needs (b)'s gold-key arm at NO LIFT. It reads LIFT (0.2078, 0.29 below the bar), so the rule cannot apply, and no S_cand comparison is needed. The general A18 hold rule ("NO LIFT on every arm") also fails, since every arm lifts.

## 4. Rerun determinism and node-fault exposure
- Rerun check:
  - The rerun of GTAGKejuQTxD ran on jpbo-099-02 (usage.run.1, 01:41).
  - Its output gendecode.json (sha256 c3eb34fc...) and decode_raw.json (96eff0ec...) are byte-identical to the debugger's three login-GH200 reruns (rnnrun, rnnrun2, rnnrun3).
  - Its config equals rnnrun's config.
  - So the debugger's determinism claim holds.
- The three ep8 genmarg decodes that ran on jpbo-028-30 were 6dI3l4J7CoBB (GPU3), VJnNjKM4mCPH (GPU2) and 0PxK9UgOfqFn (GPU2). Their finished outputs are byte-identical to the debugger's reruns (I compared the hashes).
- **Correction (exposure):** the debugger's Q2 checked only the genmarg decodes. All four verdict-bearing ep8 recognizer posterior forwards also ran on jpbo-028-30 in the same SLURM pack 2004639:
  - O1zOKML2oEhv (g_dur) on GPU0, 01:01:15-01:01:50;
  - qtDoaElsEWgZ (r30_dur) on GPU0, 01:01:52-01:02:03;
  - **e9LrruCo4AdR (gold_key) on GPU1, 01:01:49-01:02:00**;
  - **9AUpBk28Ak7F (r70_dur) on GPU1, 01:02:04-01:02:16**.
  - GPU1 is the device that produced the wrong value. The fault came at 01:01:46, and the gold_key forward started 3 s later.
  - These forwards have no internal consistency check like the re-add assert.
- Evidence against corruption:
  - The posteriors are finite and normalised.
  - Per-utterance errors show no outlier: no utterance's ep8 rate exceeds the median of the other arms by more than 0.5. There are 13, 13, 11 and 13 utterances at a rate of 0.9 or more for the four arms, with GPU1 arms like GPU0 arms.
  - gold_key's one utterance that is 0.5 or more worse at ep8 than at ep4 (8288-274162-0032, 7 phones) is a plausible phonetic variant, not garbage.
  - A hardware corruption cannot plausibly lower PER, so the LIFT classes and the verdict are robust.
- Not verified: that these four PERs, and the r70 paired row built on 9AUpBk28Ak7F, are bit-exact. What would tell: rerun those four forwards on another node and byte-compare posteriors.hdf.
- Everything else was unexposed:
  - the training (jpbo-071-09);
  - the ep1/2/4 forwards (jpbo-099-46, jpbo-056-24, jpbo-007-12);
  - the ep0 reports (keyinit, jpbo-057-13);
  - every CPU job (extract, PER, report, paired, reader), all on the login node jpbl-s02-03.

## 5. Paired rows: they finished
- They are under `work/speech_llm/sae/emc/eval_jobs/`: PairedPerDeltaJob.2wMw96BmGytD (g_dur against a17_gold_em), UnT62NW3k2M9 (r30) and neMyJu1lkoSl (r70).
- Each has output/paired_per.json and summary.txt, written 01:02. The finished.tar.gz archives date from 01:02, 01:36 and 01:36.
- A = the A17 (i) ep8 PER jobs lnBncUyLK9LE, CcGCFp2OWhSG and WbGJ6P5xrKuI, which trace to `T18RrTNTdg65/a17_*_em/epoch.008.pt`. B = the lift arm.
- 2,864 utterances and 33 speakers; the CI is the 95 % speaker-clustered bootstrap.

| Row | PER A | PER B | delta | CI | My own bootstrap (other seed) |
|---|---|---|---|---|---|
| G-dur vs gold-EM | 0.2004 | 0.1931 | -0.0073 | [-0.0093, -0.0054] | [-0.0094, -0.0053] |
| r30-dur vs r30-EM | 0.2007 | 0.2059 | +0.0051 | [+0.0034, +0.0070] | [+0.0033, +0.0069] |
| r70-dur vs r70-EM | 0.3620 | 0.2275 | -0.1344 | [-0.1397, -0.1295] | [-0.1397, -0.1295] |

- The extractor's "MISSING" was a search failure.
- Reading (report only; A18 (b) registers no margin for these rows):
  - The G-dur and r30-dur deltas have opposite signs and are below the bed's same-config seed spread: A19's M = 0.010, and A18 (a)'s ep8 seed pair differs by 0.004. The CI covers utterance sampling, not training variance. So these two rows tie at the seed-spread scale and give no evidence that MFA durations help or hurt the lift.
  - r70-dur lifts far better than r70-EM (-0.134). The two inits, however, differ in more than the duration logits: they are different 48-sub-epoch EM endpoints, with init genPER 0.442/0.466 against 0.495/0.508. This is one seed each.

## 6. What the read licenses
- All four inits use labels:
  - The gold key is the MFA-alignment majority label of each unit.
  - G-dur, r30 and r70 carry emission heads fitted on supervised gold or corrupted-gold strings.
  - The keyinit audit found MFA-fitted segment structure in the G-dur and r70-dur emission heads.
  - Only the duration logits are general knowledge.
- The brief's "phis trained with general-knowledge durations (no supervised alignments)" is therefore wrong for the three durinit arms. For the gold-key arm it holds only for the durations, since its key comes from the alignment.
- Licensed:
  - Within the S basin reached from label-derived inits, replacing MFA-fitted duration logits with general-knowledge durinit does not stop the lift of a random theta: ep8 PER 0.19-0.23, one seed per arm.
  - The A18 (c) hold rule cannot fire, so BRIDGE_KEYARMS stays on.
- Not licensed:
  - Anything about label-free phis. Every label-free phi tested for lift so far reads NO LIFT: the A14 (i) A10 restarts at 0.820-0.842, and A18 (a)'s wave phi (dec_joint) at 0.842.
  - That the stage-2 key arms will lift. The gold-key control has the right names by construction, while A20 found the selected keys' names wrong (identity 0.07-0.14). The registered proxy is the gold-key arm, but its LIFT says nothing about whether the found keys lift.
  - A clean no-segmentation claim. That rests only on the gold-key arm, which still uses alignment-derived labels.

## Strongest reason
The ep8 PERs re-derive exactly from the posteriors on all 2,864 utterances, and each sits about 0.29-0.31 under the LIFT bar. The only integrity gap is the unverified bit-exactness of four forwards on the faulty node, and that gap can only have inflated PER, not produced a false lift.
