DONE

The four-arm diagnostic is internally reproducible and compares the intended fixed endpoints. The original G4a.3 cold take-off gate remains epoch-4 dev-other greedy PER <0.50 **and** a positive own-phi speaker-matched derangement gap. This diagnostic does not measure that gap; its PER clause plainly fails.

Direct banked `per.json` arithmetic, `(S+D+I)/177275`:

| arm / epoch | S | D | I | PER | hypothesis phones |
|---|---:|---:|---:|---:|---:|
| control `lam3_tri` / 4 | 99188 | 45528 | 5152 | 0.8453983923 | 136899 |
| S3c denoise / 4 | 120960 | 24456 | 11274 | 0.8838809759 | 164093 |
| control `lam3_tri` / 8 | 120871 | 25007 | 9625 | 0.8771851643 | 161893 |
| S3c denoise / 8 | 117106 | 30369 | 8513 | 0.8799210267 | 155419 |

All four score files record 2,864 dev-other utterances, 177,275 gold phones, `drop_sil=true`, `blank_id=0`. Each hypothesis-phone count also equals `N-D+I`. The diagnostic JSON reproduces the banked S/D/I/N and PER, and the newly generated control epoch-4 diagnostic matches the previous `analysis/out/emc_hyp_inspect.lam3_tri.ep4.dev-other.json` on checked values (PER, null, mapping, correlation, entropy). Source score jobs: `GreedyPerJob.BuG1h00eKzp4`, `.2N0nHw4dL7AV`, `.Vtzp31jCDiuD`, `.KP9ADSHwxm5b`, in table order; their resolved aliases and `info` files name the same `GoldPhonesJob.ZGSp0hxyd2YP` and split. Each forward job reads the same `L15FeatureHdfJob.6ChpQYsQh1VI`. Forward `info` -> `ExtractSubmoduleCheckpointJob` `info` traces control epochs 4/8 to `PackedEmcTrainJob.byYMQmBNEpLZ/output/lam3_tri` and candidate epochs 4/8 to `ReturnnTrainingJob.K9bb4EWzhKCv`. These are fixed epochs, not a PER-picked checkpoint. The rendered training configs share data, prior, anneal/rate settings and eight-epoch length; candidate adds the registered `primary_specaug_until_epoch=4` and `time_max_width=8` masking option. The prior resolves to the same `PhoneNgramPriorJob.TRPE0D5nF3bh` table in all arms.

The diagnostic's own-unigram, per-utterance hypothesis-length-matched iid null uses five draws, seed 0, and the same PER denominator/edit convention. Null PER versus actual is control 0.862811 vs 0.845398 and candidate 0.895228 vs 0.883881 at epoch 4; control 0.894780 vs 0.877185 and candidate 0.886195 vs 0.879921 at epoch 8. Thus each stored decode outperforms that limited chance construction, but the candidate's descriptive margin is smaller at both endpoints. Each arm has its **own** null, so those margins are not a paired treatment estimate; draw SDs are not confidence intervals across utterances or speakers.

The original-alignment Hungarian mapping gives candidate PER 0.880570 at epoch 4 and 0.879763 at epoch 8, still far above the gate; controls map to 0.845393 and 0.875053. The mapping is fitted and evaluated on the same gold and is not globally optimal after realignment. Small gains cannot establish that no latent phone content or alternative mapping exists. Candidate length correlation is 0.9530/0.9370 at epochs 4/8 (control 0.9186/0.9553); this shows length tracking, not correct identity. Candidate unigram entropy is 4.319/4.641 bits (control 4.509/4.775; gold 4.833), with 36/38 versus 38/39 phones used; these are frequency summaries, not identity tests.

Both arms score better than their own within-utterance shuffles under the same banked `log_tri` table: candidate -4.1866 vs -6.3659 and -4.5943 vs -6.6424 nats/phone at epochs 4/8; control -4.7923 vs -6.9070 and -4.8008 vs -7.1692. This supports sequential regularity under that prior, not gold phone identity or acoustic content. Direct raw LM scores across arms do not isolate sequence quality because token counts and unigram compositions differ. All rows omit SIL, whereas the prior fit included SIL; the held-out 9.4689 perplexity is not an absolute comparable target here. No null draw or diagnostic gold enters training or checkpoint selection.

Strongest reason for the verdict: on the same full-split greedy PER measure, S3c is 0.03848 worse than control at the gate endpoint (0.88388 vs 0.84540), and both are far above 0.50. The diagnostic numbers are valid descriptions but do not license a content-learning or phone-identity claim.

/e/project1/spell/wu24/2026-07-13_unsupervised/reports/codex_4a_s3c_hyp_inspect_audit_2026-09-17.md
