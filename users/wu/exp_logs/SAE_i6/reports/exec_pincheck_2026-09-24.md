# exec pin check 2026-09-24 -- BLOCKED (no pin verdict)

Config: config/sae_i6_pincheck.py (new; registers out_ffmpeg_binary, pin_check.txt, ffmpeg_version.txt,
pcm_sha256.txt of data.librispeech.get_checked_ffmpeg_binary()). Console graph: 4 jobs, all CPU, no GPU:
DownloadLibriSpeechMetadataJob.n7Yd9EbtVi13, DownloadLibriSpeechCorpusJob.gWxOHrcGHjNZ,
LibriSpeechCreateBlissCorpusJob.qlLkwjdH203i (all three finished, ran locally on glukose),
FfmpegPinCheckJob.42C51BEFLzTc (ERROR).

Manager pid 1534768 started 15:44 with sae env python; it exited by itself once only the error remained.
Pin job: work/i6_experiments/users/wu/experiments/unsupervised_asr/data/ffmpeg_pin/FfmpegPinCheckJob.42C51BEFLzTc,
SLURM 4333654 (cpu_modern, cn-601, 32 s, COMPLETED 0:0 in sacct), error.run.1.

Cause: ModuleNotFoundError: No module named 'i6_core' at ffmpeg_pin.py:238 (`from i6_core.lib import corpus`
inside run()). Sisyphus runs a task inside `execute_in_dir(<job>/work)` (task.py:177), and IMPORT_PATHS are
relative ("config", "recipe", "recipe/"; global_settings.py:215, not overridden in settings.py); RecipeFinder
resolves them with os.path.abspath against the job work dir, so a lazy import of a not-yet-imported recipe
package inside run() fails. i6_experiments was importable only because it was imported while unpickling job.save.
Not a pin failure; ffmpeg never ran (no ffmpeg_version.txt, output/ empty).

Fix options (outside my edit scope, hash-neutral): absolute IMPORT_PATHS in settings.py
(e.g. f"{os.path.dirname(os.path.abspath(__file__))}/recipe/"), or move the i6_core import in
FfmpegPinCheckJob.run() (and any similar lazy run()-time imports in the package) to module level.
After the fix: clear with -co (stateless check job) and restart the manager.
