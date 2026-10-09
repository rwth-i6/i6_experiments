# exp2026_10_09_unsupervised_asr — unpaired phoneme-map search (Part II of the project)

This package holds the analysis line that grew out of the Sisyphus setup in
`../exp2026_04_09_unsupervised_asr`. It asks how to learn a map from audio to phonemes using
audio and phoneme text that are never paired, by **matching n-gram statistics** with a forward KL.

These are **plain scripts**: no Sisyphus graph and no RETURNN. They run on CPU or GPU and are
submitted with SLURM through the launchers here. They read inputs produced by the Sisyphus setup
(feature dumps, cluster and phoneme HDFs) by absolute path, under
`/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/...` and zyang's cheat-seg HDFs.

**Start with `docs/CURRENT_SETUP.md`**, which describes the most recent experiment (E.3) in full.

## Layout

```
docs/
  CURRENT_SETUP.md        the current experiment (Task E.3: continuous generator + n-gram criterion), standalone
  RESULTS.md              running results; Part I = the Sisyphus models, Part II = Tasks A-E of this line
  PLAN_TASK_E.md          plan + outcomes of Task E (emission model on real audio)
  OBJECTIVES.md           compact math summary of the mapping criteria
  OBJECTIVE_VARIANTS.md   variants of those criteria
scripts/                  all analysis scripts, which import each other as siblings
launchers/taskB..taskE/   the SLURM launchers and table helpers used for each task
```

## Scripts (`scripts/`)

| script | task | what it does |
|---|---|---|
| `calc_shuffle_control.py` | — | shuffled-reference control for any recognition output; run it first on any new output |
| `calc_output_stats_identifiability.py` | — | how much of the labeling the unigram and bigram output-statistics losses pin down |
| `calc_relabeling_search.py` | — | LM-guided relabeling of decoded output (an LM score alone is the wrong objective) |
| `calc_cheat_seg_identifiability.py` | — / C | oracle ceiling + hard-climb identifiability of the cheat-seg 512→41 map; also provides the data loaders the others use |
| `calc_embedding_isometry.py`, `calc_encoder_symbol_embeddings.py` | — | vecmap / Procrustes test (negative) |
| `calc_soft_map_search.py` | B, C | soft map + forward-KL n-gram criterion up to order 4 (`SoftMapLoss`) — the reference implementation |
| `calc_corpus_diagnostics.py` | A | geminate mass, compression ratio, Markov-order diagnostics |
| `calc_unsegmented_map.py` | C | the criterion on real clus128 tokens + collapse; supplies the PER and shuffle helpers |
| `calc_hmm_map_search.py` | D | frozen-transition HMM (full-length forward KL), EM |
| `calc_emission_ladder.py` | E.1 | supervised-ceiling ladder over audio representations; `FeatureStore`, `agglomerate` |
| `calc_e2_unsup_map.py` | E.2 | HMM-EM → order-4 criterion on a real discrete stream |
| `calc_e3_generator.py` | E.3 | Conv1d generator over continuous segments + the same criterion (GPU) |

### Running
Run from `scripts/`, or call the files by path: a sibling import works because the script's own
directory is on `sys.path`.

| use | venv |
|---|---|
| CPU scripts | `/work/asr4/schmitt/venvs/returnn_torch/bin/python3` |
| GPU (E.3) | `/work/asr4/schmitt/venvs/torch-2.11/bin/python3` (cu121) |

Examples:

```bash
cd scripts
/work/asr4/schmitt/venvs/torch-2.11/bin/python3 calc_e3_generator.py selftest
bash ../launchers/taskE/launch_e3.sh basin
```

### Where outputs go
- **Outputs and logs** go to `/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/`.
  The home directory has only a few GB, so never write large data there.
- **Paths in the docs:** relative `plan_runs/...` paths mean that directory.
- **Cheat-seg cache:** pass `--cache .../plan_runs/cheat_seg_cache.pkl`. The default under
  `/var/tmp` is node-local.

## Rules this line follows
- **Report every seed:** give the hardened loss and PER per seed, plus the seed selected by loss.
  Report **identifiability** (how close an oracle or supervised start's optimum is to the truth)
  separately from the **basin** (cold-start success rate).
- **Select by hardened loss only:** seeds, hyperparameters and stopping are chosen by hardened loss,
  never by accuracy.
- **Never flip the KL:** it stays forward, `KL(text ‖ model)`; never `KL(q ‖ p)`.
- **Leave the existing criterion and count-table code as it is,** so earlier results stay
  reproducible.

## Provenance
- **Origin:** the scripts and docs were moved here unchanged on 2026-10-09 from the root of the
  experiment directory, `/u/schmitt/experiments/2026_04_09_unsupervised_asr/`. Launchers came from
  `plan_runs/task*/`.
- **Edits made during the move** (the moved E.3 self-test still reproduces its earlier output
  exactly):
  - the launchers' `cd` now points to `scripts/`;
  - `calc_encoder_symbol_embeddings.py` hard-codes the Sisyphus setup dir for `work/` and `recipe/`
    instead of using its own location;
  - two table helpers now read logs by absolute path.
- **Old commands in the logs:** logs written before the move show the old invocation, e.g.
  `python3 calc_x.py` from the experiment root.
