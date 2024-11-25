from sisyphus import tk
from ..baseline.baseline_config import run_switchboard_baseline_ldc_v5
from i6_experiments.users.vieting.jobs.alignment import ExtractWordBoundariesFromAlignmentJob

gmm_system = run_switchboard_baseline_ldc_v5(recognition=False)
job = ExtractWordBoundariesFromAlignmentJob(
    alignment_cache=gmm_system.outputs["switchboard"]["final"].alignments.alternatives["task_dependent"],
    allophone_file=gmm_system.outputs["switchboard"]["final"].crp.acoustic_model_post_config.allophones.add_from_file,
    corpus_file=gmm_system.outputs["switchboard"]["final"].crp.corpus_config.file
)
tk.register_output("word_boundaries.xml.gz", job.out_word_boundaries)
