# Launch: packed S3b training node (2026-09-15)

Config: config/sae_4a_s3b_pack.py (imports recipe/2025-10-speech-llm configs/config_sae_4a_s3b_pack_v1.py, commit 69efa16).

sae_4a_s3b_pack is not in sis_managers.sh's IN_SCOPE list (verified by grep), so it was started
with the manual nohup form from the sis skill's Start section (venv activated first: `source
/e/project1/spell/wu24/env/sis_env/bin/activate`, confirmed `which python/black/sis` resolve into
sis_env), same pattern the earlier sae_4a_s3b_rate/lam3 launch used (that config is likewise absent
from sis_managers.sh's IN_SCOPE array).

Command:
```
cd /e/project1/spell/wu24/2026-07-13_unsupervised
source /e/project1/spell/wu24/env/sis_env/bin/activate
SIS="/e/project1/spell/wu24/env/sis_env/bin/python tools/sisyphus/sis"
NAME=sae_4a_s3b_pack
{ nohup $SIS --log_level 30 m -r "config/$NAME.py" > "log/$NAME.manager.log" 2>&1 < /dev/null & \
  echo $! > "log/$NAME.manager.pid"; }
```
No error/clear prompt was hit (fresh graph, no prior error state) -- normal start sufficed, `-io`
was not needed.

## Result
- Manager pid: 2499469 (alive, config/sae_4a_s3b_pack.py)
- SLURM job id: 1819395 (array task 1819395_1), state RUNNING (confirmed via squeue)
- Job: PackedEmcTrainJob.SeZzGUScxq4x
- Job dir: /e/scratch/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/pack_jobs/PackedEmcTrainJob.SeZzGUScxq4x
  (work/ is a symlink to this scratch path from the setup dir)
- Per-arm output subdirs confirmed on disk under job dir's output/: lam1, lam10, spec, spec_speed
- log.run.1 shows all four arms launched via returnn/rnn.py with distinct CUDA_VISIBLE_DEVICES
  (0-3) and growing RSS/VMS, i.e. actively training.
- manager.log: only one harmless pre-existing WARNING (stale HuggingFace download cleanup dir),
  no errors.

## Watcher command for the orchestrator
```
bash ~/.claude/skills/sis/sis_watch.sh 2499469 config/sae_4a_s3b_pack.py 300
```
