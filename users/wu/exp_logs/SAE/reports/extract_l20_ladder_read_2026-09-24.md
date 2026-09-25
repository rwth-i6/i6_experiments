# L2-0 ladder read extraction (2026-09-24)

No job output found anywhere under the four named job dirs, `output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_ladder/`, or `reports/` grep that literally prints the strings LIFT / PARTIAL / NO LIFT / rho*_lift / NO SEPARATING STATISTIC / CANNOT_TELL as a verdict for this ladder. The rules (LIFT<0.50, PARTIAL<0.8164 at ep8; bar-statistic test) are stated in the config docstring and in SAE_4A_lexlat_v2.md but no reader job in the four named job dirs computes/prints them; per.txt files hold raw PER only, and the disjoint-CV reader (EktNvNSRrXKj) prints only statistics (a)/(b), explicitly "No job here applies a bar, a band or a verdict."

## Job dirs (resolved)
- Reader EktNvNSRrXKj: work/speech_llm/sae/emc/blankfree_a13_disjoint/LadderCompetenceDisjointReadJob.EktNvNSRrXKj
- Pack P WX41NC734WLo: work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.WX41NC734WLo
- Pack R1 mZaZk7Ptt5Sg: work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.mZaZk7Ptt5Sg
- Pack R2 UdhhxiGIMBob: work/speech_llm/sae/emc/blankfree_pack_jobs/PackedBlankfreeTrainJob.UdhhxiGIMBob
(all under /e/scratch/spell/wu24/2026-07-13_unsupervised/work/... via the work/ symlink)

## 2. Dev-other greedy PER, per.txt, by arm/epoch
(output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_ladder/<arm>/<epN>/dev-other/per.txt)

Node R1/R2 (theta = cold random init):
| arm | ep1 | ep2 | ep4 | ep8 |
|---|---|---|---|---|
| rt_r0 (gold phi) | 0.180674 | 0.189587 | 0.204349 | 0.177634 |
| rt_r0_s2 | 0.180877 | 0.199374 | 0.188966 | 0.178576 |
| rt_r30 | 0.186123 | 0.198003 | 0.203119 | 0.182643 |
| rt_r50 | 0.190196 | 0.201190 | 0.204016 | 0.186326 |
| rt_r70 | 0.253307 | 0.217549 | 0.213087 | 0.190901 |
| rt_r100 | 0.844451 | 0.870410 | 0.861836 | 0.855169 |
| cold_ctl (random phi) | 0.888089 | 0.886374 | 0.855400 | 0.848698 |
| rt_perm (permuted-gold phi) | 0.889104 | 0.884451 | 0.867065 | 0.865508 |

Node P (anchor ladder, theta = p0):
| arm | ep1 | ep2 | ep4 | ep8 |
|---|---|---|---|---|
| corrphi_k2lat_r30 | 0.182947 | 0.198432 | 0.204180 | 0.181673 |
| corrphi_k2lat_r50 | 0.187624 | 0.199143 | 0.200903 | 0.182355 |
| corrphi_k2lat_r70 | 0.228368 | 0.215106 | 0.211011 | 0.188696 |
| corrphi_k2lat_r100 | 0.819709 | 0.876113 | 0.861086 | 0.852963 |

Each per.txt line format e.g.: "dev-other PER=0.177634 S=11713 D=16565 I=3212 N=177275" (full S/D/I/N given per arm/epoch above at the paths cited; omitted here for brevity, present verbatim in the per.txt files).

## 3. Statistics (a) and (b)

### From LadderCompetenceDisjointReadJob.EktNvNSRrXKj/output/report.txt (A13, 260-primary / 285-secondary)
Statistic (a): tau=1 NLL per frame (utterance mean; pooled) and tau=2 free energy F_2 per frame; lower = higher marginal likelihood
| phi | fit | 260 nll_tau1 (mean) | 260 nll_tau1 (pooled) | 260 F_2 | 285 nll_tau1 (mean) | 285 nll_tau1 (pooled) | 285 F_2 |
|---|---|---|---|---|---|---|---|
| gold | 2821 | 3.47351 | 3.45601 | 3.14847 | 3.47024 | 3.45600 | 3.14608 |
| r30 | 2821 | 3.71916 | 3.70680 | 3.30890 | 3.71978 | 3.70994 | 3.30958 |
| r50 | 2821 | 3.95858 | 3.94328 | 3.48843 | 3.95899 | 3.94702 | 3.48863 |
| r70 | 2821 | 4.41132 | 4.40022 | 3.79028 | 4.40844 | 4.39965 | 3.78778 |
| r100 | 2821 | 4.69008 | 4.68055 | 3.90156 | 4.68612 | 4.67873 | 3.89796 |
| phi_c | disj. | 3.49873 | 3.49107 | 3.13045 | 3.50388 | 3.49678 | 3.13606 |
| permphi | 2821 | 4.07474 | 4.05550 | 3.72216 | 4.07467 | 4.05828 | 3.72220 |

Statistic (b) (260/285), own decode minus same-speaker deranged own decode, nats/frame, mean [speaker-clustered 95% CI], matched n:
| phi | 260 | 285 (secondary) |
|---|---|---|
| gold | +4.30206 [+4.07899,+4.47178] n=159 (of 260 decoded, 101 no donor) | +4.33257 [+4.13611,+4.48334] n=190 |
| r30 | +2.52532 [+2.38045,+2.64715] n=160 (100 no donor) | +2.53218 [+2.40503,+2.64979] n=190 |
| r50 | +1.90674 [+1.79122,+2.01135] n=161 (99 no donor) | +1.91319 [+1.80915,+2.01649] n=191 |
| r70 | +1.07688 [+0.97487,+1.19927] n=162 (98 no donor) | +1.09110 [+0.99808,+1.19937] n=192 |
| r100 | +0.63798 [+0.51802,+0.79306] n=164 (96 no donor) | +0.64333 [+0.54211,+0.78039] n=194 |
| phi_c | +4.55507 [+4.30687,+4.76848] n=161 (99 no donor) | +4.55814 [+4.34652,+4.75041] n=191 |
| permphi | +4.10106 [+3.88729,+4.26279] n=160 (100 no donor) | +4.10208 [+3.91715,+4.24679] n=190 |

fit '2821' = fitted on CvHoldoutSplitJob.sD7U6CYs8ACM train.segments (25 of 285 CV-holdout utts are fit items); 'disj.' = fit excludes CV holdout.

### From output/.../sae_4a_lexlat_v2_ladder/competence/<phi>/{cv_holdout,dev-other}/ (earlier pre-A13 reads; cv_holdout = the plain 285 set, dev-other = 500-utt set)
Statistic (a), genmarg.json summary.nll_tau1 / free_energy_tau2 (pooled_per_frame):
| phi | set | nll_tau1 pooled/frame | nll_tau1 mean/frame | F_2 pooled/frame |
|---|---|---|---|---|
| gold | cv_holdout(285) | 3.4560018 | 3.4702351 | 3.1299874 |
| gold | dev-other(500) | 3.6710060 | 3.7382540 | 3.3540424 |
| r30 | cv_holdout | 3.7099375 | 3.7197793 | 3.2966607 |
| r30 | dev-other | 3.8698548 | 3.9210484 | 3.4679983 |
| r50 | cv_holdout | 3.9470189 | 3.9589895 | 3.4732039 |
| r50 | dev-other | 4.0952624 | 4.1538207 | 3.6353981 |
| r70 | cv_holdout | 4.3996547 | 4.4084352 | 3.7757255 |
| r70 | dev-other | 4.5415931 | 4.5925964 | 3.9253033 |
| r100 | cv_holdout | 4.6787270 | 4.6861169 | 3.8857337 |
| r100 | dev-other | 4.8081567 | 4.8377497 | 4.0277442 |
| phi_c | cv_holdout | 3.4967835 | 3.5038819 | 3.1282102 |
| phi_c | dev-other | 3.7016107 | 3.7228136 | 3.3366432 |
| permphi | cv_holdout | 4.0582780 | 4.0746693 | 3.7032548 |
| permphi | dev-other | 4.2606315 | 4.3423884 | 3.9162887 |

Statistic (b), gap/summary.txt (own - deranged, per_frame mean [95% CI], n matched):
| phi | cv_holdout(285) | dev-other(500) |
|---|---|---|
| gold | +4.332568 [+4.136113,+4.483341] matched 190/285 (95 no donor) | +4.306974 [+4.196677,+4.417316] matched 500/500 |
| r30 | +2.532182 [+2.405027,+2.649793] matched 190 (95 no donor) | +2.547580 [+2.495285,+2.600484] matched 500 |
| r50 | +1.913187 [+1.809146,+2.016490] matched 191 (94 no donor) | +1.892269 [+1.845949,+1.939313] matched 500 |
| r70 | +1.091105 [+0.998077,+1.199374] matched 192 (93 no donor) | +1.018396 [+0.985450,+1.055655] matched 500 |
| r100 | +0.643331 [+0.542114,+0.780392] matched 194 (91 no donor) | +0.535290 [+0.513815,+0.558253] matched 500 |
| phi_c | +4.558142 [+4.346523,+4.750414] matched 191 (94 no donor) | +4.124261 [+4.051353,+4.200390] matched 500 |
| permphi | +4.102078 [+3.917151,+4.246793] matched 190 (95 no donor) | +4.109679 [+3.995578,+4.225038] matched 500 |

Statistic (c) (label-using gold minus deranged gold, dev-other only): MISSING (searched: output/.../sae_4a_lexlat_v2_ladder/competence/*/dev-other/gap/*, LadderCompetenceDisjointReadJob output which explicitly states "dev-other statistics are unchanged and not read here"; config docstring says statistic (c) on 500 CV-holdout utterances is NOT BUILT "because its inputs do not exist"; REUSED_GAP_READS values for gold phi and D14 decphi not located in these four job dirs).

## 4. Paired rows (arm vs p0), from output/.../sae_4a_lexlat_v2_ladder/paired/<arm>_vs_p0/<ep>/dev-other/summary.txt
| arm | ep | delta_per [95% CI] | PER A(p0)->B | macro delta [CI] | improved/worse/tied | refines |
|---|---|---|---|---|---|---|
| corrphi_k2lat_r30 | ep4 | +0.014734 [+0.009436,+0.019781] | 0.189446->0.204180 | +0.009936 [+0.003329,+0.016502] | 970/1487/407 | False |
| corrphi_k2lat_r30 | ep8 | -0.007773 [-0.013172,-0.002619] | 0.189446->0.181673 | -0.014409 [-0.020900,-0.007780] | 1389/1086/389 | True |
| corrphi_k2lat_r50 | ep4 | +0.011457 [+0.006715,+0.016132] | 0.189446->0.200903 | +0.007017 [+0.001687,+0.012397] | 1044/1476/344 | False |
| corrphi_k2lat_r50 | ep8 | -0.007091 [-0.012884,-0.001409] | 0.189446->0.182355 | -0.014454 [-0.021433,-0.007491] | 1379/1084/401 | True |
| corrphi_k2lat_r70 | ep4 | +0.021565 [+0.015458,+0.027417] | 0.189446->0.211011 | +0.013787 [+0.006265,+0.020871] | 887/1628/349 | False |
| corrphi_k2lat_r70 | ep8 | -0.000750 [-0.006554,+0.004965] | 0.189446->0.188696 | -0.005499 [-0.012703,+0.001460] | 1292/1206/366 | False |
| corrphi_k2lat_r100 | ep4 | +0.671640 [+0.658732,+0.685506] | 0.189446->0.861086 | +0.682533 [+0.670451,+0.696222] | 0/2861/3 | False |
| corrphi_k2lat_r100 | ep8 | +0.663517 [+0.650806,+0.677171] | 0.189446->0.852963 | +0.670789 [+0.658893,+0.684341] | 2/2861/1 | False |

(all on 2864 utts / 33 speakers, 95% speaker-clustered bootstrap, 2000 resamples, seed 0; negative = B better)

No paired rt_* (R1/R2) vs anything found in these job dirs.

## 1. Reader verdict lines (LIFT/PARTIAL/NO LIFT, rho*_lift, bar statistic, L2-2 funding)
MISSING (searched: EktNvNSRrXKj/output/report.txt and job.save/log.run.1; output/.../sae_4a_lexlat_v2_ladder/ tree for any file containing LIFT/PARTIAL/CANNOT_TELL/rho*_lift/funding text; none of the four named job dirs computes or prints these verdicts — report.txt explicitly disclaims "No job here applies a bar, a band or a verdict.")

## 5. Output paths
- output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_ladder/{rt_r0,rt_r0_s2,rt_r30,rt_r50,rt_r70,rt_r100,cold_ctl,rt_perm}/{ep1,ep2,ep4,ep8}/dev-other/per.txt
- output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_ladder/corrphi_k2lat_{r30,r50,r70,r100}/{ep1,ep2,ep4,ep8}/dev-other/per.txt
- output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_ladder/paired/corrphi_k2lat_{r30,r50,r70,r100}_vs_p0/{ep4,ep8}/dev-other/summary.txt
- output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_ladder/competence/{gold,r30,r50,r70,r100,phi_c,permphi}/{cv_holdout,dev-other}/{genmarg.json,gap/summary.txt}
- output/exp2025_11_06_speech_llms/librispeech/sae_4a_lexlat_v2_ladder/competence_a13/report.txt (symlink) -> work/speech_llm/sae/emc/blankfree_a13_disjoint/LadderCompetenceDisjointReadJob.EktNvNSRrXKj/output/report.txt
