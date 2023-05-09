"""
Prepare data needed for hybrid systems on 960h Librispeech.
"""
import numpy as np
from sisyphus import tk
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.common.baselines.librispeech.ls960.gmm.baseline_config import run_librispeech_960_common_baseline


def get_hdf_alignments():
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="6d2945a85cc95df5349a59541d84f172dd55cc20",
    ).out_repository

    gmm_system = run_librispeech_960_common_baseline(recognition=False)
    state_tying_job = DumpStateTyingJob(gmm_system.outputs["train-other-960"]["final"].crp)
    allophone_file = gmm_system.outputs["train-other-960"]["final"].crp.acoustic_model_post_config.allophones.add_from_file
    train_align_job = RasrAlignmentDumpHDFJob(
        alignment_caches=list(gmm_system.outputs["train-other-960"]["final"].alignments.hidden_paths.values()),
        state_tying_file=state_tying_job.out_state_tying,
        allophone_file=allophone_file,
        data_type=np.int16,
        returnn_root=returnn_root,
    )
    tk.register_output("tmp", train_align_job.out_hdf_files[0])
    return None