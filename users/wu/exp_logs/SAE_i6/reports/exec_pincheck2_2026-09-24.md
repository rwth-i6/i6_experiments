# Pin check rerun 2026-09-24
Status: DONE (pin verdict FAIL; the run itself completed its check)
- settings.py: appended IMPORT_PATHS = [SETUP/config, SETUP/recipe, SETUP/recipe/] (absolute) with a comment. Checked against sisyphus/loader.py RecipeFinder: no trailing slash = dirname as search dir + basename as module prefix; trailing slash = root with no prefix. Meaning unchanged.
- Hash check (console, config/sae_i6_p0.py): ReturnnTrainingJob.zoGpJfrJw3As, ReturnnTrainingJob.ybGmGzaLx2Fn, FfmpegPinCheckJob.42C51BEFLzTc all present, so unchanged.
- Rerun: manager -co (pid 1540316, then stopped), old dir moved to FfmpegPinCheckJob.42C51BEFLzTc.cleared.0001. SLURM 4333657. The import error is fixed; ffmpeg ran.
- Verdict: FAIL, 2850 of 2864 recordings differ from banked audio (first: 116-288045-0000..0004).
- ffmpeg: /work/asr4/hwu/conda/envs/sae/bin/ffmpeg, "ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers" (same version as the reference conda-forge 7.1.1, so the difference is in the build, or in how the bank was made).
- Wall time 3:04 (136% CPU, RSS 390 MB).
- Job dir: SETUP/work/i6_experiments/users/wu/experiments/unsupervised_asr/data/ffmpeg_pin/FfmpegPinCheckJob.42C51BEFLzTc (error.run.1; outputs pcm_sha256.txt, ffmpeg_version.txt, pin_check.txt)
