# w2vu2 launch steps 1-3 (2026-09-25)

Status: DONE.

1. settings.py sha256 before = 743b4f7e...d752 (as reviewed). Backup: settings.py.pre_w2vu2_2026-09-25.
   analysis_out/w2vu2_settings.patch applied cleanly. `import settings` under the sae python works;
   W2VU_PYTHON = /work/asr4/hwu/conda/envs/w2vu/bin/w2vu-python; FairseqHydraTrainingJob run 72 h floor present (settings.py:402-406).
   Managers P0 (1646677) and P1 ladder (1726329) still alive, not restarted.
2. sbatch analysis/w2vu_env_build/build.sbatch -> Slurm 4364831, partition gpu_32gb, node cn-32, RUNNING;
   log /u/hwu/setups/librispeech-960/2026-09-24-unsupervised/log/w2vu_env_build.4364831.out (growing). Not waited on.
3. w2vu2_port_extras.sh (CONDA_BIN=/bin/false) rc=0, log log/w2vu2_port_extras_2026-09-25.log.
   Final line: "OK torch 2.7.1 torchaudio 2.7.1+cu126 sklearn 1.8.0".
   Before: torch 2.7.1, scikit-learn 1.8.0, numpy 2.4.6, no torchaudio. After: same plus torchaudio 2.7.1.
   Full `pip freeze` diff: only `+ torchaudio==2.7.1`.

Not done (orchestrator's decision): step 4, the w2vu2 manager. Gate is in the review report.
