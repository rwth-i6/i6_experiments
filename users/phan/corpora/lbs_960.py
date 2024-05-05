from typing import Union
import h5py
import shutil
import os
import copy
import numpy as np

from sisyphus import tk, Job, Task
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_experiments.common.baselines.librispeech.ls960.gmm.baseline_config import run_librispeech_960_common_baseline

class ConvertGMMAlignmentIndicesToCTCPhonemeIndices(Job):
    """
    Convert the HDF of GMM alignments with 3-state indices to
    CTC phoneme indices.
    Requirements should be similar to dump ReturnnDumpHDFJob
    """
    def __init__(
        self,
        gmm_alignments_hdf: Union[str, tk.Path],
        gmm_state_tying: Union[str, tk.Path],
        ctc_state_tying: Union[str, tk.Path],
        gmm_silence_phone="[SILENCE]",
        ctc_blank="<blank>",
        cpu=4,
        mem=16,
        file_size=1000,
        time=4,
    ):
        self.gmm_alignments_hdf = gmm_alignments_hdf
        self.gmm_state_tying = gmm_state_tying
        self.ctc_state_tying = ctc_state_tying
        self.gmm_silence_phone = gmm_silence_phone
        self.ctc_blank = ctc_blank
        self.rqmt = {
            "cpu": cpu,
            "mem": mem,
            "file_size": file_size,
            "time": time,
        }
        self.out_hdf = self.output_path("data.hdf")

    def tasks(self):
        yield Task("run", resume="run", rqmt = self.rqmt)
    
    def run(self):
        # dict mapping gmm state index -> phoneme (each phoneme 3 states in gmm)
        # line format: [SILENCE]{#+#}@i@f.0 117
        gmm_idx_to_phoneme = {}
        with open(self.gmm_state_tying, "r") as gmm_state_indices:
            for line in gmm_state_indices.readlines():
                center = line.split("{")[0]
                # Check if EOW
                if "@f" in line and center != self.gmm_silence_phone:
                    phoneme = center + "#"
                else:
                    phoneme = center
                if phoneme == self.gmm_silence_phone:
                    phoneme = self.ctc_blank
                idx = int(line.split()[-1])
                gmm_idx_to_phoneme[idx] = phoneme
        print("GMM indices to phonemes: ", gmm_idx_to_phoneme)

        # mapping phoneme -> ctc phoneme index
        # line format: AA{#+#}.0 1
        ctc_phoneme_to_idx = {}
        with open(self.ctc_state_tying, "r") as ctc_phoneme_idx:
            for line in ctc_phoneme_idx.readlines():
                phoneme = line.split("{")[0]
                idx = int(line.split()[-1])
                ctc_phoneme_to_idx[phoneme] = idx
        print("CTC phonemes to indices: ", ctc_phoneme_to_idx)

        index_mapping = {idx: ctc_phoneme_to_idx[gmm_idx_to_phoneme[idx]] for idx in gmm_idx_to_phoneme}
        print("GMM phoneme index to CTC phoneme index: ", index_mapping)

        # open the gmm alignment hdf, replace the indices and dump it
        # silence will become blank (0)
        # gmm = h5py.File(self.gmm_alignments_hdf, "r")
        temp_path = "temp_out.hdf"
        shutil.copyfile(self.gmm_alignments_hdf, temp_path)
        temp = h5py.File(temp_path, "a")
        arr = temp["/inputs"][:]
        map_func = lambda idx: index_mapping[idx]
        vec_map_func = np.vectorize(map_func) # this makes life good. for-loop does not
        arr_mapped = vec_map_func(arr)
        temp["/inputs"][:] = arr_mapped
        temp.close()
        shutil.copyfile(temp_path, self.out_hdf)
        os.remove(temp_path)


    @classmethod
    def hash(cls, parsed_args):
        del parsed_args["cpu"]
        del parsed_args["mem"]
        del parsed_args["file_size"]
        del parsed_args["time"]
        return super().hash(parsed_args)

def get_gmm_monophone_eow_state_tying():
    # system = run_librispeech_960_common_baseline()
    # crp = system.crp["train-other-960"]
    # tying_crp = copy.deepcopy(crp)
    # tying_crp.acoustic_model_config.state_tying.type = "monophone-eow"
    # st = DumpStateTyingJob(tying_crp).out_state_tying
    # tk.register_output("state_tying_monophone-eow", st)
    # return st
    # this to prevent running the search jobs
    return tk.Path("/work/asr3/zyang/share/mnphan/work_torch_ctc_libri/i6_core/lexicon/allophones/DumpStateTyingJob.ZNT27goNHX4j/output/state-tying")

def get_train_gmm_alignments_hdf(returnn_python_exe: tk.Path, returnn_root: tk.Path, train_key="train-other-960", state_tying_type="monophone-eow"):
    if train_key == "train-other-960":
        # this is a GMM alignment, need to convert to phoneme index
        if state_tying_type  == "monophone":
            gmm_alignments = tk.Path("/work/common/asr/librispeech/data/sisyphus_export_setup/alias/baselines/librispeech/ls960/gmm/common_baseline/train/train-other-960_mono_align_last/output/alignment.cache.bundle")
            # monophone state tying (but differ between 3 states of each phoneme)
            gmm_state_tying = tk.Path("/work/common/asr/librispeech/data/sisyphus_export_setup/output/baselines/librispeech/ls960/gmm/common_baseline/train-other-960_mono_state_tying")
        elif state_tying_type == "monophone-eow":
            gmm_alignments = tk.Path("/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle")
            # monophone eow state tying
            # this one does not work
            # gmm_state_tying = tk.Path("/work/asr4/raissi/setups/librispeech/960-ls/2023-01--system_paper/work/i6_core/lexicon/allophones/DumpStateTyingJob.K6yUDmjLaQIh/output/state-tying")
            gmm_state_tying = get_gmm_monophone_eow_state_tying()
        allophone = tk.Path("/work/common/asr/librispeech/data/sisyphus_export_setup/output/baselines/librispeech/ls960/gmm/common_baseline/train-other-960.allophones")
        gmm_alignments_hdf = AlignmentData(
            alignment_cache_bundle=gmm_alignments,
            allophone_file=allophone,
            state_tying_file=gmm_state_tying,
            silence_phone="[SILENCE]",
        ).get_hdf(returnn_python_exe, returnn_root)
        # okay I need to learn to work with h5py
        ctc_state_tying = tk.Path("/u/minh-nghia.phan/setups/torch/librispeech/work/i6_core/lexicon/allophones/DumpStateTyingJob.vfz3NEyDiM05/output/state-tying")
        gmm_alignments_hdf = ConvertGMMAlignmentIndicesToCTCPhonemeIndices(
            gmm_alignments_hdf,
            gmm_state_tying,
            ctc_state_tying,
        ).out_hdf
    else:
        raise NotImplementedError(f"GMM alignments for train key {train_key} is not avaibale")
    return gmm_alignments_hdf

if __name__ == "__main__":
    gmm_alignments_correct_index_hdf = ConvertGMMAlignmentIndicesToCTCPhonemeIndices(
        "/u/minh-nghia.phan/setups/torch/librispeech/work/i6_core/returnn/hdf/ReturnnDumpHDFJob.cCXJImS6K7Vg/output/data.hdf",
        "/work/common/asr/librispeech/data/sisyphus_export_setup/output/baselines/librispeech/ls960/gmm/common_baseline/train-other-960_mono_state_tying",
        "/u/minh-nghia.phan/setups/torch/librispeech/work/i6_core/lexicon/allophones/DumpStateTyingJob.vfz3NEyDiM05/output/state-tying",
    ).run()
