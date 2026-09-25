# A15-E extract, 2026-09-24

## Job presence check
21/21 PhiEmissionMapJob dirs present under
work/speech_llm/sae/emc/phi_emission_map/ (resolved:
/e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_emission_map/):
PhiEmissionMapJob.47dOkByqkih4, .6ogcjGS3iWlZ, .76tjFPjGaNUO, .A7Mw7BT4I4uV, .agYgeieCNYdE,
.B2hk4qG76odL, .cL2CpcVWkoCl, .IUQNOtGb5FRf, .kO72ftoy6ON5, .MbZVwuaQVw8y, .NqkaNumrJYZk,
.Oo0OaKbeoT5h, .OOiR1dVPrDj9, .PIWNc6yZ0HY6, .sgZQ7zYGoZ5j, .SRejeTst9zEb, .UkLyXfCDmyZ7,
.WcJ3Bk3gfwmx, .WkAszZze9nuG, .YeU0B7TL7ywa, .ZJdVMGqFKsnp.
Aggregate reader: PhiEmissionMapReadJob.mZWCsgaj92GS, has `finished` marker and
output/table.txt + output/table.json (Sep 24 04:16). None missing.
Slurm: maps were routed via gpupack jobs 1989528 (16 maps) and 1989530 (5 maps),
per log/gpupack/jobs/1989528/ and .../1989530/ rc/start files (both present, matching
the launch report). Reader submit_log.run shows engine_info jpbl-s02-03 (short/local engine).

## Aggregate reader's printed table (verbatim from output/table.txt)

A15-E emission-matched map -- DESCRIPTIVE, label-using, NO GATE, NEVER SELECTS
set: D4 dev-other, 500 selected; item set sizes [500]; T columns: per-frame nats, first string minus second; the sign reads per column (REPORTING RULES below: T3 relabel is about labelling, not a preference for gold)

phi                      | CORRECT | JS dir mat  | UNCL | T3 primary [95% CI]      | T3 relabel [95% CI]      | R4 acc dir mat (oracle)
a10_uniform_s01_ep48     | -       | 0.832 0.532 |   15 | +1.989 [+1.892,+2.084]   | +1.486 [+1.409,+1.556]   | 0.112 0.309 (0.615)
a10_uniform_s02_ep48     | -       | 0.821 0.549 |   16 | +2.008 [+1.926,+2.092]   | +1.225 [+1.143,+1.296]   | 0.110 0.291 (0.615)
a10_durinit_s01_ep48     | -       | 0.862 0.511 |   11 | +2.199 [+2.117,+2.280]   | +1.696 [+1.619,+1.780]   | 0.093 0.312 (0.615)
a10_durinit_s02_ep48     | -       | 0.792 0.529 |   19 | +2.020 [+1.952,+2.094]   | +1.409 [+1.350,+1.467]   | 0.119 0.290 (0.615)
a10_durfrz_s01_ep48      | -       | 0.868 0.531 |   16 | +2.031 [+1.941,+2.122]   | +1.565 [+1.488,+1.648]   | 0.074 0.294 (0.615)
a10_durfrz_s02_ep48      | -       | 0.802 0.498 |   14 | +1.997 [+1.912,+2.082]   | +1.582 [+1.491,+1.673]   | 0.113 0.303 (0.615)
a10_durinit_s01_ep4      | -       | 0.855 0.549 |   14 | +1.772 [+1.690,+1.857]   | +1.427 [+1.330,+1.516]   | 0.080 0.291 (0.615)
a10_durinit_s01_ep12     | -       | 0.866 0.518 |   12 | +2.075 [+2.000,+2.150]   | +1.408 [+1.335,+1.471]   | 0.080 0.288 (0.615)
a10_durfrz_s01_ep4       | -       | 0.857 0.567 |   16 | +1.769 [+1.661,+1.883]   | +1.299 [+1.185,+1.411]   | 0.090 0.291 (0.615)
a10_durfrz_s01_ep12      | -       | 0.868 0.543 |   15 | +1.919 [+1.832,+2.006]   | +1.463 [+1.385,+1.546]   | 0.081 0.286 (0.615)
gold                     | 40/40   | 0.000 0.000 |    0 | +3.929 [+3.795,+4.054]   | +0.000 [+0.000,+0.000]   | 0.589 0.589 (0.615)
r30                      | -       | 0.116 0.116 |    0 | +2.218 [+2.152,+2.283]   | +0.000 [+0.000,+0.000]   | 0.571 0.571 (0.615)
r50                      | -       | 0.232 0.232 |    0 | +1.584 [+1.525,+1.640]   | +0.000 [+0.000,+0.000]   | 0.548 0.548 (0.615)
r70                      | -       | 0.532 0.522 |   15 | +0.587 [+0.552,+0.621]   | -0.044 [-0.057,-0.031]   | 0.371 0.342 (0.615)
r100                     | -       | 0.728 0.652 |   37 | +0.178 [+0.162,+0.196]   | -0.014 [-0.032,+0.001]   | 0.092 0.136 (0.615)
permphi                  | 40/40   | 0.868 0.007 |    0 | +3.924 [+3.784,+4.057]   | +4.762 [+4.610,+4.923]   | 0.071 0.591 (0.615)
phi_c                    | -       | 0.827 0.563 |   15 | +1.747 [+1.681,+1.814]   | +1.074 [+0.996,+1.155]   | 0.099 0.284 (0.615)
decphi                   | -       | 0.025 0.025 |    0 | +4.087 [+3.983,+4.189]   | +0.000 [+0.000,+0.000]   | 0.583 0.583 (0.615)
random_init              | -       | 0.717 0.716 |   39 | -0.001 [-0.006,+0.005]   | +0.004 [+0.003,+0.005]   | 0.080 0.080 (0.615)
rt_r0_ep8                | -       | 0.052 0.052 |    0 | +4.416 [+4.298,+4.522]   | +0.000 [+0.000,+0.000]   | 0.591 0.591 (0.615)
rt_r70_ep8               | -       | 0.111 0.111 |    0 | +4.039 [+3.922,+4.152]   | +0.000 [+0.000,+0.000]   | 0.564 0.564 (0.615)

unclaimed gold types: a10_uniform_s01_ep48 AE,AO,AW,ER,G,IH,JH,L,NG,OY,SH,TH,UH,V,ZH; a10_uniform_s02_ep48 AA,AH,AO,AW,B,CH,EH,ER,G,IH,JH,OY,SH,TH,UH,ZH; a10_durinit_s01_ep48 AE,AO,AW,B,EH,F,K,OY,SH,TH,UH; a10_durinit_s02_ep48 AE,AH,AO,AW,B,D,EH,F,G,IH,JH,NG,OY,R,TH,UH,UW,V,ZH; a10_durfrz_s01_ep48 AA,AE,AO,AW,F,G,IH,JH,NG,OY,R,SH,TH,UH,UW,ZH; a10_durfrz_s02_ep48 AE,AH,AO,AW,EH,EY,G,JH,OY,SH,UH,UW,V,ZH; a10_durinit_s01_ep4 AE,AO,AW,B,EH,EY,F,IY,K,NG,OY,SH,TH,UH; a10_durinit_s01_ep12 AA,AE,AO,AW,B,F,IH,K,OY,SH,TH,UH; a10_durfrz_s01_ep4 AA,AE,AH,AO,AW,EH,F,G,JH,NG,OY,P,TH,UH,UW,ZH; a10_durfrz_s01_ep12 AA,AE,AO,AW,F,G,IH,JH,NG,OY,P,SH,UH,UW,ZH; gold -; r30 -; r50 -; r70 AA,AE,AW,AY,CH,DH,F,G,JH,OY,R,TH,UH,UW,Y; r100 AA,AE,AO,AW,AY,B,CH,D,DH,EH,ER,EY,F,G,HH,IH,IY,JH,K,L,M,N,NG,OW,OY,P,S,SH,T,TH,UH,UW,V,W,Y,Z,ZH; permphi -; phi_c AE,AW,EY,F,JH,NG,OW,OY,P,R,SH,TH,UW,Y,ZH; decphi -; random_init AA,AE,AO,AW,AY,B,CH,D,DH,EH,ER,EY,F,G,HH,IH,IY,JH,K,L,M,N,NG,OW,OY,P,R,S,SH,T,TH,UH,UW,V,W,Y,Z,ZH,SIL; rt_r0_ep8 -; rt_r70_ep8 -

R4 non-SIL frames (direct / matched): a10_uniform_s01_ep48 0.055/0.281, a10_uniform_s02_ep48 0.053/0.256, a10_durinit_s01_ep48 0.034/0.277, a10_durinit_s02_ep48 0.082/0.272, a10_durfrz_s01_ep48 0.029/0.264, a10_durfrz_s02_ep48 0.077/0.285, a10_durinit_s01_ep4 0.028/0.260, a10_durinit_s01_ep12 0.030/0.270, a10_durfrz_s01_ep4 0.037/0.255, a10_durfrz_s01_ep12 0.030/0.256, gold 0.581/0.581, r30 0.559/0.559, r50 0.534/0.534, r70 0.340/0.310, r100 0.035/0.085, permphi 0.016/0.583, phi_c 0.047/0.252, decphi 0.577/0.577, random_init 0.000/0.000, rt_r0_ep8 0.578/0.578, rt_r70_ep8 0.547/0.547

R4 oracle: all 0.6154, non-SIL 0.6104 over 130685 frames

## REPORTING RULES and CONVENTIONS (verbatim, from output/table.txt)

REPORTING RULES:
, verbatim from `SAE_4A_lexlat_v2.md` amendment A15-E (added 2026-09-24, before
    any battery result) and A15:

        "disclosed label-using analysis, descriptive, no gate"
        "The emission map: a one-to-one Hungarian assignment of the phi's 40 symbols to gold-phi
        phones, on the per-phone JS cost between R3's mean unit distributions. Also reported, per
        symbol, the many-to-one nearest gold phone with its JS, and how many gold phones are claimed
        by no symbol."
        "Through the emission map: T3 (primary, relabel), R3 matched and R4 matched."
        "Positive control: the emission map recovers permphi's true permutation (count of correct
        labels). The gold phi maps to the identity."
        "Descriptive, no gate, never selects. Rules go in the job docstring."

    So: DESCRIPTIVE, NO GATE, NEVER SELECTS.  No row is ranked, thresholded or chosen here or from
    here; the table prints every phi in the registered order (the A15 battery's).  Columns:
    CORRECT = the emission map's correct labels of 40 (known only for permphi and gold, "-"
    otherwise); JS dir / mat = mean JS (bits) over the 39 non-SIL phones, identity labelling / through
    the emission map (JS mat IS R3 matched); UNCL = gold types (of 40) claimed by no symbol as its
    nearest; T3 PRIMARY = M(gold) - M(deranged) and T3 RELABEL = M(gold) - gold, M the emission map,
    each the speaker-clustered bootstrap 95 % interval of the per-frame mean, with the count dropped
    for infeasible relabelled strings appended; R4 = frame accuracy over all retained frames, direct
    (identity labelling) / matched (through the emission map), with the majority-unit oracle ceiling.
    SIGN (per-frame nats, first string minus second): T3 PRIMARY positive = under the emission
    labelling phi prefers the utterance's own content; T3 RELABEL positive = phi fits the emission
    labelling better than the identity labelling (a statement about labelling, not a preference for
    gold).  The column conventions are ``PhiEmissionMapJob``'s (``emc.phi_emission_map`` module
    docstring), printed at the foot of the table.

CONVENTIONS:
* m_phi(u | k) and m_gold(u | k) = the battery's R3 mean unit distributions
  (``phi_competence_battery.mean_unit_distribution``: the emission averaged over duration and
  position buckets with the frame weights of the phi's own duration law, then over the etas of the
  battery's item set, utterance-uniform), for all 40 types (the 39 phones and SIL) of the phi and of
  the gold phi.  COST[s, k] = JS(m_phi(. | s), m_gold(. | k)) in bits (``js_bits``), a 40 x 40 matrix.
* EMISSION MAP h = the one-to-one assignment of the phi's 40 symbols to the gold phi's 40 types
  minimising the summed COST (``scipy.optimize.linear_sum_assignment``); h^-1(k) = the symbol assigned
  to gold type k.  SIL is a type of both phis, so gold SIL takes the symbol assigned to gold SIL (no
  DELETE convention is needed, unlike R1's decode map).
* NEAREST (many-to-one): per symbol s, argmin_k COST[s, k] and that JS.  UNCLAIMED = the gold types
  (of 40, SIL included) that are the nearest type of no symbol; listed by name.
* JS DIRECT = mean over the 39 non-SIL phones of COST[k, k] (the identity labelling; the battery's R3
  DIRECT); JS MATCHED = mean over the 39 non-SIL phones of COST[h^-1(k), k] (R3 MATCHED through the
  emission map).  SIL and the all-40 means are in the json.
* CORRECT LABELS = the number of gold types k (of 40) with h^-1(k) equal to the known symbol of k:
  permphi's own permutation (``PermuteSeedGoldJob`` ``permutation.json``, SIL fixed) or the identity
  (the gold phi); unknown (None) for every other phi.
* T1 = the battery's T1 (gold minus the same-speaker deranged gold string, the battery's item set
  and pairing); it is computed because the T3 contrasts read it.
* T3 through the emission map = ``PhiCompetenceBatteryJob.t3_contrasts`` with M = h^-1: PRIMARY =
  M(gold) minus M(deranged) (the same derangement as T1), RELABEL = M(gold) minus gold, MIXED =
  M(gold) minus T1's deranged string; per-frame utterance means with the speaker-clustered bootstrap
  95 % interval (2000 resamples, seed 0), on the utterances whose M(gold) and M(deranged) are both
  feasible (the dropped count is reported).
* R4 = frame accuracy of the unit -> phone map u -> argmax_k m_phi(u | g(k)) pi(k) (pi = the bed
  prior's unigram) against the MFA label of every retained frame of the selected utterances (the
  battery's ``MfaFrameLabelsJob`` labels), all frames and non-SIL frames.  DIRECT: g = identity (the
  battery's R4 DIRECT); MATCHED: g = h^-1, the emission map.  The ceiling is the same majority-unit
  oracle (``MfaFrameLabelsJob``'s ``oracle.json``).

## Per-phi emission-map correct-label count (from output/table.json, field emission_map.correct_labels)
- gold: 40 (of 40) -- work/.../PhiEmissionMapReadJob.mZWCsgaj92GS/output/table.json
- permphi: 40 (of 40) -- same file
- all other 19 phis (a10 x10, r30, r50, r70, r100, phi_c, decphi, random_init, rt_r0_ep8, rt_r70_ep8): null (not defined; reported "-" in the table, per convention "unknown (None) for every other phi")

T3 CI per-phi "dropped" (infeasible relabelled strings) field in table.json (t3.*.dropped): {} (empty dict) for every one of the 21 phis -- i.e. the json's t3 sub-object carries no populated "dropped" keys in this read; the "count dropped for infeasible relabelled strings" mentioned in REPORTING RULES is not separately itemized per-phi beyond what is in table.txt's T3 columns (which report only the interval, no explicit drop count printed in table.txt either).

## Many-to-one nearest-phone summary for the 10 A10 EM phis (from table.json emission_map.nearest, symbol -> nearest gold phone; computed by counting, over the 40 symbols, how many claim each gold phone)
- a10_uniform_s01_ep48 (6-restart, sub-ep48): 40 symbols -> 25 distinct gold phones claimed; most-claimed: T(4), N(3), SIL(3), HH(2), W(2)
- a10_uniform_s02_ep48 (ep48): 40 -> 24 claimed; most-claimed: SIL(3), N(3), W(3), HH(2), D(2)
- a10_durinit_s01_ep48 (ep48): 40 -> 29 claimed; most-claimed: Z(3), DH(2), W(2), SIL(2), IY(2)
- a10_durinit_s02_ep48 (ep48): 40 -> 21 claimed; most-claimed: AY(4), SIL(4), N(4), W(3), T(3)
- a10_durfrz_s01_ep48 (ep48): 40 -> 24 claimed; most-claimed: SIL(3), AY(3), Z(3), D(2), W(2)
- a10_durfrz_s02_ep48 (ep48): 40 -> 26 claimed; most-claimed: SIL(3), M(3), N(3), K(2), B(2)
- a10_durinit_s01_ep4 (trajectory ep4): 40 -> 26 claimed; most-claimed: W(3), SIL(3), AY(3), N(3), DH(2)
- a10_durinit_s01_ep12 (trajectory ep12): 40 -> 28 claimed; most-claimed: DH(2), D(2), W(2), SIL(2), Z(2)
- a10_durfrz_s01_ep4 (trajectory ep4): 40 -> 24 claimed; most-claimed: N(3), W(3), AY(3), Z(3), T(3)
- a10_durfrz_s01_ep12 (trajectory ep12): 40 -> 25 claimed; most-claimed: W(3), SIL(3), AY(3), Z(3), N(2)
(unclaimed gold types per phi are listed verbatim in the table.txt block above)

## Job IDs and output paths
- 21 PhiEmissionMapJob dirs: work/speech_llm/sae/emc/phi_emission_map/PhiEmissionMapJob.<hash>/
  (resolved: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_emission_map/PhiEmissionMapJob.<hash>/),
  hashes listed above under "Job presence check". Each has finished.tar.gz.
- Aggregate reader: work/speech_llm/sae/emc/phi_emission_map/PhiEmissionMapReadJob.mZWCsgaj92GS/
  (resolved: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_emission_map/PhiEmissionMapReadJob.mZWCsgaj92GS/)
  output/table.txt, output/table.json.
- Slurm allocation (from launch report and log/gpupack/jobs/): 1989528 (gpupack.le1h.16t, 16 maps packed one node) and 1989530 (gpupack.le1h.5t, 5 maps packed one node); reader job engine "short"/local, engine_info jpbl-s02-03 (submit_log.run).

## Completeness check
All 21 PhiEmissionMapJob dirs present; none missing. Reader (aggregate) present and finished, all 21 phis appear in table.txt/table.json `order`/`rows`.
