from typing import List
import numpy as np
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.returnn.hdf import RasrAlignmentDumpHDFJob
from i6_experiments.common.baselines.librispeech.ls960.gmm.baseline_config import (
    run_librispeech_960_common_baseline,
)
from sisyphus import tk


def get_alignment_hdf(returnn_root: tk.Path) -> List[tk.Path]:
    gmm_system = run_librispeech_960_common_baseline()

    state_tying_job = DumpStateTyingJob(gmm_system.outputs["train-other-960"]["final"].crp)
    allophone_file = gmm_system.outputs["train-other-960"][
        "final"
    ].crp.acoustic_model_post_config.allophones.add_from_file  # type: ignore
    train_align_job = RasrAlignmentDumpHDFJob(
        alignment_caches=list(
            gmm_system.outputs["train-other-960"]["final"].alignments.hidden_paths.values()  # type: ignore
        ),
        state_tying_file=state_tying_job.out_state_tying,
        allophone_file=allophone_file,
        data_type=np.int16,
        returnn_root=returnn_root,
    )
    return train_align_job.out_hdf_files
