from sisyphus import tk, Path

from ...setups.common.hdf import RasrForcedTriphoneAlignmentToHDF

from .config import ZHOU_ALLOPHONES, ZHOU_SUBSAMPLED_ALIGNMENT


def run():
    tying = Path(
        "/work/asr3/raissi/shared_workspaces/gunz/dependencies/state-tying/no-tying-dense-with-zhou-contextless-allophones/state-tying"
    )
    with open(tying, "rt") as f:
        lines = [l for l in f if not l.startswith("#") and len(l.strip()) > 0]
        num_tied_classes = len(lines)

    a_job = RasrForcedTriphoneAlignmentToHDF(
        alignment_bundle=Path(ZHOU_SUBSAMPLED_ALIGNMENT, cached=True),
        allophones=Path(ZHOU_ALLOPHONES),
        state_tying=tying,
        num_tied_classes=num_tied_classes,  # in ^
    )
    a_job.add_alias("alignments/zhou-forced-triphone")

    tk.register_output("alignments/zhou-forced-triphone/alignment.hdf", a_job.out_hdf_file)
    tk.register_output("alignments/zhou-forced-triphone/segments.lit", a_job.out_segments)

    return a_job.out_hdf_file
