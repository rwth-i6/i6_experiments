# A15 phi competence battery extraction (2026-09-24)

Reader: work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryReadJob.HWlC9YZ1U2GI/output/table.txt (and table.json)

All 21 phis present: order has 21 entries, paths has 21 entries, rows has 21 entries.

## 1. table.txt verbatim (includes docstring/convention lines)

```
A15 phi competence battery -- DESCRIPTIVE, label-using, NO GATE, NEVER SELECTS
set: D4 dev-other, 500 selected; item set sizes [500]; T columns: per-frame nats, first string minus second; the sign reads per column (REPORTING RULES below: T3 relabel is about labelling, not a preference for gold)

phi                      | T1 [95% CI]              | T2 pair1 mean [range]    | T2 pair5 mean [range]    | T2 full mean [range]     | T3 primary [95% CI]      | T3 relabel [95% CI]      | T3 mixed [95% CI]        | T3own prim relab mixed | R1 PERd PERh NMI E[d]    | R2 PERd PERh NMI   | R3 JSd JSm  | R4 acc dir mat (oracle)
a10_uniform_s01_ep48     | +0.107 [+0.059,+0.158]   | +0.041 [+0.006,+0.117]   | +0.215 [+0.063,+0.397]   | +0.425 [+0.295,+0.514]   | +0.090 [+0.046,+0.138]   | -0.425 [-0.466,-0.382]   | -0.318 [-0.374,-0.261]   | -                      | 0.849 0.861 0.086  6.19   | 0.861 0.869 0.084 | 0.832 0.831 | 0.112 0.051 (0.615)    
a10_uniform_s02_ep48     | +0.288 [+0.243,+0.338]   | +0.047 [-0.044,+0.150]   | +0.251 [-0.002,+0.435]   | +0.710 [+0.668,+0.751]   | +0.271 [+0.232,+0.318]   | -0.432 [-0.473,-0.390]   | -0.144 [-0.189,-0.094]   | -                      | 0.833 0.844 0.102  6.30   | 0.843 0.852 0.101 | 0.821 0.818 | 0.110 0.055 (0.615)    
a10_durinit_s01_ep48     | +0.078 [+0.032,+0.126]   | +0.006 [-0.007,+0.029]   | +0.172 [+0.102,+0.221]   | +0.347 [+0.309,+0.413]   | +0.100 [+0.054,+0.150]   | -0.277 [-0.304,-0.249]   | -0.199 [-0.256,-0.138]   | -                      | 0.853 0.861 0.079  6.01   | 0.857 0.864 0.084 | 0.862 0.838 | 0.093 0.053 (0.615)    
a10_durinit_s02_ep48     | +0.341 [+0.290,+0.394]   | +0.016 [-0.079,+0.084]   | +0.165 [-0.035,+0.276]   | +0.566 [+0.447,+0.638]   | +0.412 [+0.361,+0.466]   | -0.013 [-0.036,+0.012]   | +0.328 [+0.271,+0.386]   | -                      | 0.831 0.838 0.113  6.21   | 0.841 0.846 0.117 | 0.792 0.777 | 0.119 0.122 (0.615)    
a10_durfrz_s01_ep48      | +0.116 [+0.078,+0.156]   | +0.037 [-0.004,+0.074]   | +0.144 [+0.081,+0.184]   | +0.418 [+0.326,+0.557]   | +0.225 [+0.183,+0.269]   | -0.072 [-0.099,-0.043]   | +0.044 [-0.003,+0.096]   | -                      | 0.856 0.857 0.078  4.41   | 0.860 0.860 0.083 | 0.868 0.808 | 0.074 0.059 (0.615)    
a10_durfrz_s02_ep48      | +0.247 [+0.212,+0.286]   | +0.041 [+0.003,+0.093]   | +0.147 [-0.004,+0.268]   | +0.538 [+0.368,+0.646]   | +0.249 [+0.209,+0.292]   | -0.109 [-0.124,-0.094]   | +0.138 [+0.102,+0.176]   | -                      | 0.839 0.848 0.098  4.41   | 0.848 0.855 0.100 | 0.802 0.801 | 0.113 0.113 (0.615)    
a10_durinit_s01_ep4      | +0.150 [+0.099,+0.205]   | +0.002 [-0.062,+0.039]   | +0.180 [+0.051,+0.249]   | +0.378 [+0.209,+0.666]   | +0.214 [+0.161,+0.272]   | -0.129 [-0.155,-0.102]   | +0.022 [-0.038,+0.083]   | -                      | 0.862 0.869 0.068  4.86   | 0.870 0.870 0.068 | 0.855 0.843 | 0.080 0.058 (0.615)    
a10_durinit_s01_ep12     | +0.077 [+0.037,+0.123]   | +0.012 [-0.003,+0.027]   | +0.159 [+0.083,+0.223]   | +0.351 [+0.275,+0.440]   | +0.085 [+0.044,+0.130]   | -0.535 [-0.582,-0.484]   | -0.458 [-0.519,-0.389]   | -                      | 0.853 0.863 0.079  5.62   | 0.860 0.866 0.075 | 0.866 0.846 | 0.080 0.031 (0.615)    
a10_durfrz_s01_ep4       | +0.230 [+0.163,+0.300]   | +0.051 [-0.037,+0.186]   | +0.232 [+0.132,+0.311]   | +0.648 [+0.338,+1.072]   | +0.257 [+0.190,+0.328]   | -0.082 [-0.101,-0.064]   | +0.148 [+0.076,+0.220]   | -                      | 0.854 0.862 0.073  4.41   | 0.861 0.859 0.075 | 0.857 0.813 | 0.090 0.068 (0.615)    
a10_durfrz_s01_ep12      | +0.115 [+0.072,+0.163]   | +0.047 [-0.009,+0.087]   | +0.163 [+0.076,+0.210]   | +0.491 [+0.368,+0.671]   | +0.154 [+0.105,+0.204]   | -0.351 [-0.393,-0.307]   | -0.236 [-0.297,-0.171]   | -                      | 0.856 0.860 0.077  4.41   | 0.861 0.859 0.079 | 0.868 0.820 | 0.081 0.038 (0.615)    
gold                     | +3.929 [+3.795,+4.054]   | +0.280 [+0.077,+0.459]   | +1.518 [+1.139,+1.719]   | +4.465 [+4.332,+4.629]   | +3.929 [+3.795,+4.054]   | +0.000 [+0.000,+0.000]   | +3.929 [+3.795,+4.054]   | -                      | 0.193 0.193 0.835  5.11   | 0.276 0.276 0.804 | 0.000 0.000 | 0.589 0.589 (0.615)    
r30                      | +2.218 [+2.152,+2.283]   | +0.134 [+0.037,+0.208]   | +0.748 [+0.621,+0.847]   | +2.440 [+2.344,+2.500]   | +2.218 [+2.152,+2.283]   | +0.000 [+0.000,+0.000]   | +2.218 [+2.152,+2.283]   | -                      | 0.240 0.240 0.801  4.84   | 0.363 0.363 0.753 | 0.116 0.116 | 0.571 0.571 (0.615)    
r50                      | +1.584 [+1.525,+1.640]   | +0.092 [+0.027,+0.158]   | +0.529 [+0.452,+0.592]   | +1.773 [+1.678,+1.840]   | +1.584 [+1.525,+1.640]   | +0.000 [+0.000,+0.000]   | +1.584 [+1.525,+1.640]   | -                      | 0.313 0.313 0.736  4.63   | 0.437 0.437 0.702 | 0.232 0.232 | 0.548 0.548 (0.615)    
r70                      | +0.570 [+0.539,+0.602]   | +0.032 [+0.006,+0.048]   | +0.196 [+0.173,+0.207]   | +0.636 [+0.594,+0.686]   | +0.571 [+0.540,+0.601]   | -0.146 [-0.160,-0.133]   | +0.425 [+0.396,+0.454]   | -                      | 0.608 0.628 0.410  4.28   | 0.635 0.648 0.492 | 0.532 0.533 | 0.371 0.295 (0.615)    
r100                     | +0.000 [-0.013,+0.016]   | +0.002 [-0.000,+0.006]   | +0.011 [+0.003,+0.020]   | +0.055 [+0.015,+0.097]   | +0.008 [-0.004,+0.024]   | -0.175 [-0.187,-0.161]   | -0.174 [-0.191,-0.156]   | -                      | 0.826 0.830 0.112  4.22   | 0.866 0.862 0.114 | 0.728 0.729 | 0.092 0.085 (0.615)    
permphi                  | -0.071 [-0.108,-0.036]   | +0.021 [-0.022,+0.059]   | -0.072 [-0.134,+0.041]   | -0.239 [-0.310,-0.204]   | +0.806 [+0.758,+0.852]   | +1.126 [+1.055,+1.201]   | +1.055 [+0.986,+1.128]   | +3.924 +4.762 +4.691   | 0.911 0.819 0.056  5.11   | 0.920 0.605 0.060 | 0.868 0.507 | 0.071 0.182 (0.615)    
phi_c                    | +0.161 [+0.124,+0.203]   | +0.010 [-0.035,+0.032]   | +0.241 [+0.092,+0.417]   | +0.522 [+0.422,+0.752]   | +0.275 [+0.237,+0.315]   | -0.266 [-0.306,-0.221]   | -0.105 [-0.157,-0.049]   | -                      | 0.858 0.868 0.076  5.39   | 0.859 0.859 0.079 | 0.827 0.796 | 0.099 0.072 (0.615)    
decphi                   | +4.087 [+3.983,+4.189]   | +0.278 [+0.077,+0.447]   | +1.505 [+1.171,+1.706]   | +4.577 [+4.414,+4.770]   | +4.087 [+3.983,+4.189]   | +0.000 [+0.000,+0.000]   | +4.087 [+3.983,+4.189]   | -                      | 0.210 0.210 0.831  5.31   | 0.310 0.310 0.798 | 0.025 0.025 | 0.583 0.583 (0.615)    
random_init              | -0.001 [-0.006,+0.005]   | -0.000 [-0.000,+0.000]   | -0.000 [-0.000,+0.000]   | -0.000 [-0.000,-0.000]   | -0.001 [-0.006,+0.005]   | -0.004 [-0.004,-0.003]   | -0.004 [-0.009,+0.001]   | -                      | 0.938 0.903 0.401 13.50   | 0.987 0.921 0.214 | 0.717 0.717 | 0.080 0.080 (0.615)    
rt_r0_ep8                | +4.416 [+4.298,+4.522]   | +0.302 [+0.084,+0.493]   | +1.630 [+1.221,+1.864]   | +4.910 [+4.753,+5.105]   | +4.416 [+4.298,+4.522]   | +0.000 [+0.000,+0.000]   | +4.416 [+4.298,+4.522]   | -                      | 0.207 0.207 0.820  5.02   | 0.292 0.292 0.779 | 0.052 0.052 | 0.591 0.591 (0.615)    
rt_r70_ep8               | +4.039 [+3.922,+4.152]   | +0.269 [+0.044,+0.434]   | +1.420 [+1.105,+1.713]   | +4.325 [+4.197,+4.429]   | +4.039 [+3.922,+4.152]   | +0.000 [+0.000,+0.000]   | +4.039 [+3.922,+4.152]   | -                      | 0.228 0.228 0.801  4.89   | 0.326 0.326 0.742 | 0.111 0.111 | 0.564 0.564 (0.615)    

R4 non-SIL frames (direct / matched): a10_uniform_s01_ep48 0.055/0.055, a10_uniform_s02_ep48 0.053/0.056, a10_durinit_s01_ep48 0.034/0.038, a10_durinit_s02_ep48 0.082/0.090, a10_durfrz_s01_ep48 0.029/0.048, a10_durfrz_s02_ep48 0.077/0.077, a10_durinit_s01_ep4 0.028/0.033, a10_durinit_s01_ep12 0.030/0.029, a10_durfrz_s01_ep4 0.037/0.045, a10_durfrz_s01_ep12 0.030/0.041, gold 0.581/0.581, r30 0.559/0.559, r50 0.534/0.534, r70 0.340/0.243, r100 0.035/0.014, permphi 0.016/0.190, phi_c 0.047/0.075, decphi 0.577/0.577, random_init 0.000/0.000, rt_r0_ep8 0.578/0.578, rt_r70_ep8 0.547/0.547
R4 oracle: all 0.6154, non-SIL 0.6104 over 130685 frames

REPORTING RULES:
, verbatim from `SAE_4A_lexlat_v2.md` amendment A15 (user request 2026-09-24;
    registered before any job):

        "disclosed label-using analysis, descriptive, no gate"
        "It compares the reverse models directly, apart from any recognizer."
        "Text preference. Each is the per-frame log p_phi(z | string, eta), marginalised over
        segmentations: gold minus the alternative. Speaker-clustered 95 % intervals."
        "(T2) [...] 5 permutation seeds each, reported as a mean and a range."
        "(T3) [...] Amended 2026-09-24, before any result, because that contrast mixes the
        relabelling with the utterance swap. The primary T3 is now M(gold) against M(deranged), the
        same map applied to both strings, so it isolates content. Also reported: M(gold) against
        gold, which reads the relabelling alone, and the original mixed contrast."
        "(R4) [...] If no alignment exists, this is reported as not built."
        "Reporting rules sit in the job docstring. The battery explains the lift and non-lift reads
        (L2-0, A14 (i)) and says what EM is missing. It never selects or gates."

    So: DESCRIPTIVE, NO GATE, NEVER SELECTS.  No row is ranked, thresholded or chosen here or from
    here; the table prints every phi in the registered order.  T3 is three columns, each with its
    speaker-clustered interval: T3 PRIMARY = M(gold) - M(deranged) (M = the phi's R1 Hungarian map,
    applied to both strings; the primary T3), T3 RELABEL = M(gold) - gold, T3 MIXED = M(gold) - T1's
    deranged string (the original registered contrast); all three on one item set, with the count
    dropped for infeasible relabelled strings appended.  T3own (permphi only) = the same three with
    its own permutation as M (means only; intervals in the json).  Intervals: T1 and T3 are the
    speaker-clustered bootstrap 95 % interval of the per-frame mean; T2 is the mean over its seeds
    and the [min, max] range of the per-seed means.  R1 / R2 are point reads of one decode each
    (PER direct / Hungarian, NMI(symbol, phone)); R3 is the mean JS (bits) over the 39 non-SIL phones,
    direct / after the Hungarian match; R4 is frame accuracy over all retained frames, direct
    (identity labelling) / matched (through R1's Hungarian map, as R1 and R3), with the majority-unit
    oracle ceiling beside them.  SIGN, per column (every T column is per-frame nats, first string
    minus second): T1 gold - deranged, positive = phi prefers the utterance's own string over another
    utterance's; T2 gold - label-permuted gold, positive = phi prefers the true labels to permuted
    ones; T3 PRIMARY M(gold) - M(deranged), positive = under its own labelling phi prefers the
    utterance's own content; T3 RELABEL M(gold) - gold, positive = phi fits its Hungarian labelling
    better than the identity labelling (a statement about labelling, not about preferring gold);
    T3 MIXED M(gold) - deranged, both effects together.  The column conventions are ``PhiCompetenceBatteryJob``'s
    (``emc.phi_competence_battery`` module docstring), printed at the foot of the table.

CONVENTIONS:
* SCORE = ``reverse.evaluate``'s per-utterance log p_phi(z | y, eta), the explicit-duration HSMM
  marginalised over segmentations (the scoring call of ``BlankfreeDecodeGapJob``); the gold string is
  ``s1a_job.with_edge_sil`` of the GoldPhonesJob phones.  A contrast is ``gold - alternative`` per
  utterance divided by the utterance's masked frame count, averaged over utterances; its 95 %
  interval is the speaker-clustered bootstrap (``d8_admission.cluster_bootstrap``, 2000 resamples,
  seed 0, the campaign's paired-read constants) -- ``BlankfreeDecodeGapJob.column``' per-frame row.
* ITEM SET = the gold-feasible utterances of the selection that have a same-speaker donor under
  ``s1a_job.build_derangement`` over the GOLD strings in sorted-tag order (``BlankfreeDecodeGapJob``'s
  gold pairing).  Feasibility depends on the reverse topology only, so the item set is the same for
  every phi and every contrast is paired across phis.
* T1 = gold minus the same-speaker deranged gold string.  For the gold phi it reproduces the banked
  ``BlankfreeDecodeGapJob.tD8bSgyBTQB8`` ``gold_minus_deranged_gold`` per-frame mean when the pairing
  is the banked one (all 500 selected utterances matched there).
* T2 = gold minus gold with the 39 non-SIL phone labels permuted (SIL fixed), at three levels
  (:data:`T2_LEVELS`): one random transposition, five disjoint random transpositions, and a full
  random permutation (``numpy.random.default_rng(seed).permutation(39)``, the rule of permphi's
  ``PermuteSeedGoldJob``); :data:`T2_SEEDS` per level.  A permutation of phone labels keeps every
  string's feasibility (the phone cap D is the same for every non-SIL type).  Reported per level as
  the mean over seeds and the range [min, max] of the per-seed means; per-seed intervals are in the
  json.
* T3 (amended 2026-09-24, before any result).  M = the relabelling: every token of a string (edge
  SIL included) replaced by the phi symbol that R1's Hungarian map sends to that phone; the map is a
  bijection of the 40 symbols onto the 39 phones plus DELETE, and gold SIL takes the symbol mapped to
  DELETE (the identity labelling's SIL -> DELETE).  M is applied to gold and to T1's deranged string
  alike.  Three contrasts, each the per-frame utterance mean with the speaker-clustered interval:
  PRIMARY = M(gold) minus M(deranged) (the same map on both strings, so it isolates content);
  RELABEL = M(gold) minus gold (the relabelling alone); MIXED = M(gold) minus T1's deranged string
  (the registered original, which mixes the relabelling with the utterance swap).  All three are
  read on ONE item set: the utterances whose M(gold) and M(deranged) are both feasible; the others
  are dropped from T3 only and counted.  With ``permutation`` given (permphi) T3_own = the same three
  contrasts with M = its own permutation (SIL fixed; always feasible).
* R1 / R2 = ``GenDecodeReportJob``'s direct PER, Hungarian PER and NMI(symbol, phone) of the phi's
  genmarg decode under the bed's phone trigram (R1) and under :class:`UniformPhonePriorJob`'s prior
  (R2); the decode's own ``settings.prior_npz`` is asserted.  E[d] = ``expected_durations`` (the A10
  diagnostics' reader), mean over the 39 phone types.
* R3 = per type k, phi's mean unit distribution m_phi(u | k) = the emission nu_phi(u | k, c, j, eta)
  averaged over duration bucket c and position bucket j with the FRAME weights of phi's own duration
  law (sum_d p(d | k) x #frames of a d-frame segment in bucket j, over d in bucket c) and then
  averaged over the etas of the item set (utterance-uniform); JS divergence in bits against the gold
  phi's m, computed the same way.  DIRECT = JS(m_phi(. | k), m_gold(. | k)); MATCHED =
  JS(m_phi(. | h^-1(k)), m_gold(. | k)) with h R1's Hungarian map (SIL <-> DELETE, as in T3).  The
  table reports the mean over the 39 non-SIL phones; per-type values are in the json.
* R4 = frame accuracy of the unit -> phone map u -> argmax_k m_phi(u | k) pi(k) (pi = the bed
  prior's unigram, ``PhoneNgramPriorJob.RtzbESkOedsT``) against the MFA label of every retained
  frame of the selected utterances; all frames and non-SIL frames.  DIRECT = the identity symbol
  labelling (symbol k read as phone k); MATCHED = through R1's Hungarian map h, as R1 and R3 are:
  u -> argmax_k m_phi(u | h^-1(k)) pi(k), h^-1(k) the symbol h sends to phone k (SIL <-> DELETE, as
  in T3).  The ceiling is :class:`MfaFrameLabelsJob`'s majority-unit oracle on the same frames.

```

## 2. Units / normalisation / sign convention (copied from table.txt REPORTING RULES / CONVENTIONS)

- T1, T2, T3 are per-frame nats: 'gold minus the alternative' per utterance divided by the utterance's masked frame count, averaged over utterances (per-utterance mean of a per-frame quantity). Intervals: speaker-clustered bootstrap 95% (2000 resamples, seed 0).
- T1: no separate per-utterance length normalisation beyond the per-utterance division by masked frame count described above.
- T2: mean over seeds + [min,max] range of per-seed means; per-seed intervals are in json only.
- T3: three columns (primary/relabel/mixed), each per-frame utterance mean with speaker-clustered interval; permphi has T3_own (means only in table.txt; intervals in json).
- R1/R2: point reads (PER direct, PER Hungarian, NMI(symbol,phone)) of one decode; E[d]=mean expected duration over 39 phone types.
- R3: JS divergence in bits, mean over 39 non-SIL phones, direct / matched (Hungarian).
- R4: frame accuracy (fraction of frames correct) over all retained frames, direct / matched (Hungarian), oracle ceiling beside them; also reported for non-SIL frames only.
- Sign convention (verbatim): T1 gold-deranged, positive = phi prefers own string over another utterance's; T2 gold-label-permuted, positive = phi prefers true labels; T3 PRIMARY M(gold)-M(deranged), positive = phi prefers own content under its own labelling; T3 RELABEL M(gold)-gold, positive = phi's Hungarian labelling fits better than identity labelling (a labelling statement, not a gold preference); T3 MIXED M(gold)-deranged, both effects combined.

## 3. Per-phi detail (from table.json)

| phi | job path (battery.json) | T1 mean [ci95] n | T2 pair1/pair5/full mean [range] | T3 primary/relabel/mixed mean [ci95] (dropped_infeasible) | R1 PERdirect/PERhung/NMI/E[d] | R2 PERdirect/PERhung/NMI | R3 JSdirect/JSmatched | R4 acc_all dir/matched (oracle_all) |
|---|---|---|---|---|---|---|---|---|
| a10_uniform_s01_ep48 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.7R1wUyYGv0PU/output/battery.json | 0.1070 [0.0590,0.1582] n=500 | 0.0412[0.0060,0.1174] / 0.2148[0.0634,0.3971] / 0.4255[0.2951,0.5139] | 0.0903[0.0455,0.1380] / -0.4254[-0.4661,-0.3823] / -0.3184[-0.3741,-0.2609] (dropped=0) | 0.8487/0.8607/0.0860/6.1866 | 0.8609/0.8690/0.0836 | 0.8316/0.8315 | 0.1124/0.0512 (oracle=0.6154) |
| a10_uniform_s02_ep48 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.3gHwetyvjQGy/output/battery.json | 0.2879 [0.2430,0.3384] n=500 | 0.0467[-0.0441,0.1501] / 0.2510[-0.0024,0.4352] / 0.7103[0.6676,0.7506] | 0.2714[0.2320,0.3178] / -0.4321[-0.4734,-0.3901] / -0.1442[-0.1885,-0.0945] (dropped=0) | 0.8325/0.8436/0.1024/6.2956 | 0.8429/0.8520/0.1015 | 0.8207/0.8184 | 0.1097/0.0545 (oracle=0.6154) |
| a10_durinit_s01_ep48 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.tVQhyykSMJUh/output/battery.json | 0.0780 [0.0323,0.1264] n=500 | 0.0060[-0.0068,0.0288] / 0.1723[0.1015,0.2210] / 0.3473[0.3087,0.4131] | 0.1003[0.0536,0.1503] / -0.2769[-0.3036,-0.2487] / -0.1989[-0.2559,-0.1379] (dropped=0) | 0.8533/0.8615/0.0788/6.0118 | 0.8571/0.8642/0.0836 | 0.8622/0.8383 | 0.0927/0.0535 (oracle=0.6154) |
| a10_durinit_s02_ep48 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.5dkMuPNyjQhQ/output/battery.json | 0.3408 [0.2898,0.3941] n=500 | 0.0163[-0.0791,0.0845] / 0.1654[-0.0348,0.2759] / 0.5658[0.4471,0.6377] | 0.4122[0.3612,0.4665] / -0.0125[-0.0360,0.0125] / 0.3282[0.2709,0.3862] (dropped=0) | 0.8315/0.8380/0.1131/6.2055 | 0.8410/0.8458/0.1167 | 0.7924/0.7766 | 0.1190/0.1215 (oracle=0.6154) |
| a10_durfrz_s01_ep48 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.ZWRYLiArogS9/output/battery.json | 0.1164 [0.0776,0.1557] n=500 | 0.0373[-0.0035,0.0741] / 0.1435[0.0813,0.1837] / 0.4176[0.3258,0.5568] | 0.2252[0.1828,0.2692] / -0.0725[-0.0988,-0.0433] / 0.0439[-0.0029,0.0956] (dropped=0) | 0.8564/0.8570/0.0784/4.4138 | 0.8598/0.8598/0.0828 | 0.8680/0.8082 | 0.0735/0.0592 (oracle=0.6154) |
| a10_durfrz_s02_ep48 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.U1C74XjFf7uT/output/battery.json | 0.2475 [0.2122,0.2864] n=500 | 0.0406[0.0028,0.0926] / 0.1468[-0.0040,0.2677] / 0.5385[0.3683,0.6461] | 0.2486[0.2094,0.2922] / -0.1092[-0.1244,-0.0940] / 0.1383[0.1025,0.1764] (dropped=0) | 0.8392/0.8478/0.0984/4.4138 | 0.8483/0.8551/0.0995 | 0.8020/0.8007 | 0.1127/0.1133 (oracle=0.6154) |
| a10_durinit_s01_ep4 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.8SXbqazRKzdw/output/battery.json | 0.1502 [0.0989,0.2053] n=500 | 0.0025[-0.0619,0.0394] / 0.1795[0.0514,0.2494] / 0.3779[0.2093,0.6663] | 0.2139[0.1606,0.2721] / -0.1286[-0.1547,-0.1024] / 0.0217[-0.0381,0.0829] (dropped=0) | 0.8624/0.8690/0.0675/4.8557 | 0.8697/0.8698/0.0679 | 0.8546/0.8432 | 0.0805/0.0581 (oracle=0.6154) |
| a10_durinit_s01_ep12 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.9UPnGdgdLvOW/output/battery.json | 0.0770 [0.0369,0.1228] n=500 | 0.0122[-0.0025,0.0274] / 0.1589[0.0833,0.2231] / 0.3514[0.2754,0.4402] | 0.0855[0.0443,0.1304] / -0.5346[-0.5820,-0.4838] / -0.4576[-0.5191,-0.3892] (dropped=0) | 0.8533/0.8627/0.0785/5.6156 | 0.8605/0.8664/0.0751 | 0.8663/0.8465 | 0.0799/0.0307 (oracle=0.6154) |
| a10_durfrz_s01_ep4 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.Q54cbbLddZck/output/battery.json | 0.2301 [0.1631,0.3001] n=500 | 0.0510[-0.0367,0.1863] / 0.2320[0.1320,0.3105] / 0.6478[0.3384,1.0718] | 0.2572[0.1902,0.3279] / -0.0819[-0.1012,-0.0635] / 0.1482[0.0760,0.2204] (dropped=0) | 0.8541/0.8617/0.0730/4.4138 | 0.8606/0.8587/0.0748 | 0.8566/0.8133 | 0.0899/0.0680 (oracle=0.6154) |
| a10_durfrz_s01_ep12 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.9aJjXO4v5H8j/output/battery.json | 0.1151 [0.0724,0.1625] n=500 | 0.0474[-0.0091,0.0873] / 0.1627[0.0759,0.2104] / 0.4911[0.3680,0.6714] | 0.1542[0.1055,0.2040] / -0.3506[-0.3927,-0.3068] / -0.2355[-0.2972,-0.1713] (dropped=0) | 0.8558/0.8598/0.0771/4.4138 | 0.8609/0.8594/0.0791 | 0.8684/0.8201 | 0.0814/0.0385 (oracle=0.6154) |
| gold | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.70VGC2j3WEqN/output/battery.json | 3.9285 [3.7946,4.0545] n=500 | 0.2802[0.0770,0.4591] / 1.5177[1.1386,1.7185] / 4.4649[4.3317,4.6293] | 3.9285[3.7946,4.0545] / 0.0000[0.0000,0.0000] / 3.9285[3.7946,4.0545] (dropped=0) | 0.1933/0.1933/0.8350/5.1064 | 0.2760/0.2760/0.8043 | 0.0000/0.0000 | 0.5895/0.5895 (oracle=0.6154) |
| r30 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.GD5Pe8xyfcM1/output/battery.json | 2.2185 [2.1518,2.2825] n=500 | 0.1341[0.0365,0.2076] / 0.7480[0.6206,0.8471] / 2.4403[2.3436,2.4996] | 2.2185[2.1518,2.2825] / 0.0000[0.0000,0.0000] / 2.2185[2.1518,2.2825] (dropped=0) | 0.2404/0.2404/0.8013/4.8444 | 0.3632/0.3632/0.7533 | 0.1163/0.1163 | 0.5712/0.5712 (oracle=0.6154) |
| r50 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.lHYhC9QeUMG2/output/battery.json | 1.5837 [1.5247,1.6403] n=500 | 0.0919[0.0266,0.1581] / 0.5295[0.4519,0.5925] / 1.7725[1.6783,1.8398] | 1.5837[1.5247,1.6403] / 0.0000[0.0000,0.0000] / 1.5837[1.5247,1.6403] (dropped=0) | 0.3128/0.3128/0.7355/4.6280 | 0.4367/0.4367/0.7016 | 0.2318/0.2318 | 0.5483/0.5483 (oracle=0.6154) |
| r70 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.8gyku6WD0BH6/output/battery.json | 0.5704 [0.5394,0.6018] n=500 | 0.0315[0.0059,0.0483] / 0.1964[0.1733,0.2068] / 0.6365[0.5945,0.6856] | 0.5709[0.5397,0.6015] / -0.1458[-0.1596,-0.1329] / 0.4246[0.3963,0.4538] (dropped=0) | 0.6083/0.6282/0.4101/4.2790 | 0.6348/0.6481/0.4922 | 0.5321/0.5334 | 0.3706/0.2954 (oracle=0.6154) |
| r100 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.bgLlbNgO9m5a/output/battery.json | 0.0004 [-0.0127,0.0165] n=500 | 0.0024[-0.0001,0.0063] / 0.0109[0.0031,0.0196] / 0.0551[0.0148,0.0971] | 0.0085[-0.0043,0.0236] / -0.1745[-0.1872,-0.1614] / -0.1741[-0.1906,-0.1561] (dropped=0) | 0.8261/0.8304/0.1124/4.2198 | 0.8656/0.8625/0.1138 | 0.7278/0.7292 | 0.0921/0.0847 (oracle=0.6154) |
| permphi | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.wzVuQCn8Docf/output/battery.json | -0.0707 [-0.1080,-0.0361] n=500 | 0.0209[-0.0216,0.0588] / -0.0722[-0.1337,0.0406] / -0.2393[-0.3103,-0.2044] | 0.8055[0.7578,0.8521] / 1.1255[1.0551,1.2006] / 1.0548[0.9858,1.1278] (dropped=0) | 0.9107/0.8186/0.0559/5.1081 | 0.9203/0.6045/0.0599 | 0.8678/0.5072 | 0.0709/0.1816 (oracle=0.6154) |
| permphi T3own | (own-permutation, means; intervals in json) | primary=3.9236 relabel=4.7621 mixed=4.6914 dropped=0 | | | | | | |
| phi_c | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.afJ4n3XoFFrL/output/battery.json | 0.1607 [0.1235,0.2025] n=500 | 0.0100[-0.0349,0.0324] / 0.2413[0.0918,0.4169] / 0.5220[0.4218,0.7523] | 0.2746[0.2374,0.3152] / -0.2659[-0.3060,-0.2206] / -0.1052[-0.1568,-0.0488] (dropped=0) | 0.8578/0.8680/0.0760/5.3914 | 0.8591/0.8594/0.0786 | 0.8273/0.7961 | 0.0991/0.0715 (oracle=0.6154) |
| decphi | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.U5A7dK7Dvq8z/output/battery.json | 4.0873 [3.9827,4.1892] n=500 | 0.2782[0.0770,0.4470] / 1.5053[1.1713,1.7056] / 4.5773[4.4136,4.7702] | 4.0873[3.9827,4.1892] / 0.0000[0.0000,0.0000] / 4.0873[3.9827,4.1892] (dropped=0) | 0.2100/0.2100/0.8311/5.3128 | 0.3095/0.3095/0.7976 | 0.0248/0.0248 | 0.5835/0.5835 (oracle=0.6154) |
| random_init | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.dv6gxNYwC5FF/output/battery.json | -0.0007 [-0.0059,0.0047] n=500 | -0.0000[-0.0000,0.0000] / -0.0000[-0.0001,0.0000] / -0.0001[-0.0001,-0.0000] | -0.0009[-0.0061,0.0045] / -0.0036[-0.0043,-0.0029] / -0.0043[-0.0094,0.0012] (dropped=0) | 0.9384/0.9025/0.4013/13.5000 | 0.9870/0.9211/0.2139 | 0.7166/0.7166 | 0.0800/0.0800 (oracle=0.6154) |
| rt_r0_ep8 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.YGVHWr38jGTn/output/battery.json | 4.4161 [4.2984,4.5225] n=500 | 0.3023[0.0841,0.4926] / 1.6296[1.2212,1.8641] / 4.9097[4.7527,5.1046] | 4.4161[4.2984,4.5225] / 0.0000[0.0000,0.0000] / 4.4161[4.2984,4.5225] (dropped=0) | 0.2073/0.2073/0.8202/5.0231 | 0.2921/0.2921/0.7793 | 0.0524/0.0524 | 0.5913/0.5913 (oracle=0.6154) |
| rt_r70_ep8 | /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.nh8qTTJyBYcB/output/battery.json | 4.0392 [3.9216,4.1522] n=500 | 0.2685[0.0436,0.4342] / 1.4204[1.1054,1.7132] / 4.3254[4.1973,4.4292] | 4.0392[3.9216,4.1522] / 0.0000[0.0000,0.0000] / 4.0392[3.9216,4.1522] (dropped=0) | 0.2282/0.2282/0.8007/4.8855 | 0.3261/0.3261/0.7420 | 0.1109/0.1109 | 0.5635/0.5635 (oracle=0.6154) |

## 4. Note on T1 fraction-positive
T1 json fields per phi are only: mean, median, ci95, n, speakers. No 'fraction of utterances with positive difference' field exists in table.json for any phi -> MISSING (searched: table.json rows.*.T1, table.txt).

## 5. Job ids / paths (battery.json per phi, all 21 present)

- a10_uniform_s01_ep48: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.7R1wUyYGv0PU/output/battery.json
- a10_uniform_s02_ep48: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.3gHwetyvjQGy/output/battery.json
- a10_durinit_s01_ep48: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.tVQhyykSMJUh/output/battery.json
- a10_durinit_s02_ep48: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.5dkMuPNyjQhQ/output/battery.json
- a10_durfrz_s01_ep48: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.ZWRYLiArogS9/output/battery.json
- a10_durfrz_s02_ep48: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.U1C74XjFf7uT/output/battery.json
- a10_durinit_s01_ep4: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.8SXbqazRKzdw/output/battery.json
- a10_durinit_s01_ep12: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.9UPnGdgdLvOW/output/battery.json
- a10_durfrz_s01_ep4: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.Q54cbbLddZck/output/battery.json
- a10_durfrz_s01_ep12: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.9aJjXO4v5H8j/output/battery.json
- gold: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.70VGC2j3WEqN/output/battery.json
- r30: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.GD5Pe8xyfcM1/output/battery.json
- r50: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.lHYhC9QeUMG2/output/battery.json
- r70: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.8gyku6WD0BH6/output/battery.json
- r100: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.bgLlbNgO9m5a/output/battery.json
- permphi: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.wzVuQCn8Docf/output/battery.json
- phi_c: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.afJ4n3XoFFrL/output/battery.json
- decphi: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.U5A7dK7Dvq8z/output/battery.json
- random_init: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.dv6gxNYwC5FF/output/battery.json
- rt_r0_ep8: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.YGVHWr38jGTn/output/battery.json
- rt_r70_ep8: /e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/phi_competence_battery/PhiCompetenceBatteryJob.nh8qTTJyBYcB/output/battery.json